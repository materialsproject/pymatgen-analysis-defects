"""Classes and methods related to thermodynamics and energy."""

from __future__ import annotations

import collections
import logging
from dataclasses import dataclass, field
from itertools import chain, groupby
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
from matplotlib import pyplot as plt
from monty.dev import deprecated
from monty.json import MSONable
from pymatgen.analysis.chempot_diagram import ChemicalPotentialDiagram
from pymatgen.analysis.defects.core import Defect, NamedDefect
from pymatgen.analysis.defects.corrections.freysoldt import get_freysoldt_correction
from pymatgen.analysis.defects.finder import DefectSiteFinder
from pymatgen.analysis.defects.supercells import get_closest_sc_mat
from pymatgen.analysis.defects.utils import get_zfile, group_docs
from pymatgen.analysis.phase_diagram import PhaseDiagram
from pymatgen.analysis.structure_matcher import ElementComparator, StructureMatcher
from pymatgen.core import Composition, Element
from pymatgen.electronic_structure.dos import FermiDos
from pymatgen.entries.computed_entries import ComputedEntry
from pymatgen.io.vasp import Locpot, Vasprun, VolumetricData
from pyrho.charge_density import get_volumetric_like_sc
from scipy.constants import value as _cd
from scipy.optimize import bisect
from scipy.spatial import ConvexHull

if TYPE_CHECKING:
    from collections.abc import Generator, Sequence

    from matplotlib.axes import Axes
    from numpy.typing import ArrayLike, NDArray
    from pandas import DataFrame
    from pymatgen.analysis.defects.utils import CorrectionResult
    from pymatgen.core import Structure
    from pymatgen.electronic_structure.dos import Dos
    from pymatgen.entries.computed_entries import ComputedStructureEntry

__author__ = "Jimmy-Xuan Shen, Danny Broberg, Shyam Dwaraknath"
__copyright__ = "Copyright 2022, The Materials Project"
__maintainer__ = "Jimmy-Xuan Shen"
__email__ = "jmmshn@gmail.com"

_logger = logging.getLogger(__name__)
boltzman_eV_K = _cd("Boltzmann constant in eV/K")


@dataclass
class DefectEntry(MSONable):
    """Data for completed defect supercell calculation.

    Attributes:
        defect:
            The defect object used to generate the supercell.
        charge_state:
            The charge state of the defect.
        sc_entry:
            The ComputedStructureEntry for the supercell.
        sc_defect_frac_coords:
            The fractional coordinates of the defect in the supercell.
            If None, structures attributes of the locpot file will be used to
            automatically determine the defect location.
        bulk_entry:
            The ComputedEntry for the bulk material. If this is provided, the energy difference
            can be calculated from this automatically. The `bulk_entry` can also be provided as
            an attribute of the `DefectEntry` object. If both are provided, the one for
            `FormationEnergyDiagram` has precedence.
        corrections:
            A dictionary of corrections to the energy.
        corrections_metadata:
            A dictionary that acts as a generic container for storing information
            about how the corrections were calculated.  These should are only used
            for debugging and plotting purposes.
            PLEASE DO NOT USE THIS AS A CONTAINER FOR IMPORTANT DATA.
        entry_id:
            The entry_id for the defect entry. Usually the same as the entry_id of the
            defect supercell entry.
    """

    defect: Defect
    charge_state: int
    sc_entry: ComputedStructureEntry
    corrections: dict[str, float] = field(default_factory=dict)
    corrections_metadata: dict[str, Any] = field(default_factory=dict)
    sc_defect_frac_coords: tuple[float, float, float] | None = None
    bulk_entry: ComputedEntry | None = None
    entry_id: str | None = None

    def __post_init__(self) -> None:
        """Post-initialization."""
        self.charge_state = int(self.charge_state)
        self.entry_id = self.entry_id or self.sc_entry.entry_id

    def get_freysoldt_correction(
        self,
        defect_locpot: Locpot | dict,
        bulk_locpot: Locpot | dict,
        dielectric: float | NDArray,
        defect_struct: Structure | None = None,
        bulk_struct: Structure | None = None,
        **kwargs,
    ) -> CorrectionResult:
        """Calculate the Freysoldt correction.

        Updates the corrections dictionary with the Freysoldt correction
        and returns the planar averaged potential data for plotting.

        Args:
            defect_locpot:
                The Locpot object for the defect supercell.
                Or a dictionary of the planar averaged locpot
            bulk_locpot:
                The Locpot object for the bulk supercell.
                Or a dictionary of the planar averaged locpot
            dielectric:
                The dielectric tensor or constant for the bulk material.
            defect_struct:
                The defect structure. If None, the structure of the defect_locpot
                will be used.
            bulk_struct:
                The bulk structure. If None, the structure of the bulk_locpot
                will be used.
            **kwargs:
                Additional keyword arguments for the get_correction method.

        Returns:
            dict:
                The plotting data to analyze the planar averaged electrostatic potential
                in the three periodic lattice directions.
        """
        if defect_struct is None:
            defect_struct = getattr(defect_locpot, "structure", None)
        if bulk_struct is None:
            bulk_struct = getattr(bulk_locpot, "structure", None)

        if defect_struct is None or bulk_struct is None:  # pragma: no cover
            msg = "defect_struct and/or bulk_struct is missing either provide the structure or provide the complete locpot."
            raise ValueError(
                msg,
            )

        if self.sc_defect_frac_coords is None:
            finder = DefectSiteFinder()
            defect_fpos = finder.get_defect_fpos(
                defect_structure=defect_struct,
                base_structure=bulk_struct,
            )
            self.sc_defect_frac_coords = defect_fpos
        else:  # pragma: no cover
            defect_fpos = self.sc_defect_frac_coords

        if isinstance(defect_locpot, VolumetricData):
            defect_gn = defect_locpot.dim
        elif isinstance(defect_locpot, dict):
            defect_gn = tuple(map(len, (defect_locpot for k in ["0", "1", "2"])))

        if isinstance(bulk_locpot, VolumetricData):
            bulk_sc_locpot = get_sc_locpot(
                uc_locpot=bulk_locpot,
                defect_struct=defect_struct,
                grid_out=defect_gn,
                up_sample=2,
            )
            bulk_locpot = bulk_sc_locpot

        frey_corr = get_freysoldt_correction(
            q=self.charge_state,
            dielectric=dielectric,
            defect_locpot=defect_locpot,
            bulk_locpot=bulk_sc_locpot,
            defect_frac_coords=defect_fpos,
            lattice=defect_struct.lattice,
            **kwargs,
        )
        self.corrections.update(
            {
                "freysoldt": frey_corr.correction_energy,
            },
        )
        self.corrections_metadata.update({"freysoldt": frey_corr.metadata.copy()})
        return frey_corr

    @property
    def corrected_energy(self) -> float:
        """The energy of the defect entry with all corrections applied."""
        return self.sc_entry.energy + sum(self.corrections.values())

    def get_ediff(self) -> float | None:
        """Get the energy difference between the defect and the bulk (including finite-size correction)."""
        if self.bulk_entry is None:
            msg = (
                "Attempting to compute the energy difference without a bulk entry data."
            )
            raise RuntimeError(
                msg,
            )
        return self.corrected_energy - self.bulk_entry.energy

    def get_summary_dict(self) -> dict:
        """Get a summary dictionary for the defect entry."""
        corrections_d = {f"correction_{k}": v for k, v in self.corrections.items()}
        res = {
            "name": self.defect.name,
            "charge_state": self.charge_state,
            "bulk_total_energy": self.bulk_entry.energy if self.bulk_entry else None,
            "defect_total_energy": self.sc_entry.energy,
        }
        res.update(corrections_d)
        return res

    @property
    def defect_chemsys(self) -> str:
        """Get the chemical system of the defect."""
        return "-".join(
            sorted({el.symbol for el in self.defect.defect_structure.elements})
        )


@dataclass
class FormationEnergyDiagram(MSONable):
    """Formation energy.

    Attributes:
        defect_entries:
            The list of defect entries for the different charge states.
            The finite-size correction should already be applied to these.
        pd_entries:
            The list of entries used to construct the phase diagram and chemical
            potential diagram. They will be used to determine the stability region
            of the bulk crystal.  The entries are used instead of the ``PhaseDiagram``
            object to make serializing the object easier.
        vbm:
            The VBM of the bulk crystal.
        band_gap:
            The band gap of the bulk crystal.
        bulk_entry:
            The bulk computed entry to get the total energy of the bulk supercell.
            This is only used in case where the bulk entry data is not provided by
            the individual defect entries themselves. Default is None.
            The data from the `bulk_entry` attached to each `defect_entry` will be
            preferentially used if it is available.
        inc_inf_values:
            If False these boundary points at infinity are ignored when we look at the
            chemical potential limits.
            The stability region is sometimes unbounded, example:
            Mn_Ga in GaN, the chemical potential of Mn is does not affect
            the stability of GaN so it can go to ``-inf``.
            A artificial value is needed to help the half-space intersection algorithm.
            This can be justified since these tend to be the substitutional elements
            which should not have very negative chemical potential.
        bulk_stability:
            If the bulk energy is above the convex hull, lower it to this value below
            the convex hull.
    """

    defect_entries: list[DefectEntry]
    pd_entries: list[ComputedEntry]
    vbm: float
    band_gap: float | None = None
    bulk_entry: ComputedStructureEntry | None = None
    inc_inf_values: bool = False
    bulk_stability: float = 0.001

    def __post_init__(self) -> None:
        """Post-initialization.

        - Reconstruct the phase diagram with the bulk entry
        - Make sure that the bulk entry is stable
        - create the chemical potential diagram using only the formation energies
        """
        g = group_defect_entries(self.defect_entries)
        if next(g, True) and next(g, False):
            msg = (
                "Defects are not of same type! "
                "Use MultiFormationEnergyDiagram for multiple defect types"
            )
            raise ValueError(
                msg,
            )
        # if all of the `DefectEntry` objects have the same `bulk_entry` then `self.bulk_entry` is not needed
        if self.bulk_entry is None and any(
            x.bulk_entry is None for x in self.defect_entries
        ):
            msg = "The bulk entry must be provided."
            raise RuntimeError(
                msg,
            )

        bulk_entry = self.bulk_entry or min(
            [x.bulk_entry for x in self.defect_entries],
            key=lambda x: x.energy_per_atom,
        )
        pd_ = PhaseDiagram(self.pd_entries)
        entries = pd_.stable_entries | {bulk_entry}
        pd_ = PhaseDiagram(entries)
        self.phase_diagram = ensure_stable_bulk(pd_, bulk_entry, self.bulk_stability)
        entries = []
        for entry in self.phase_diagram.stable_entries:
            d_ = {
                "energy": self.phase_diagram.get_form_energy(entry),
                "composition": entry.composition,
                "entry_id": entry.entry_id,
                "correction": 0.0,
            }
            entries.append(ComputedEntry.from_dict(d_))
            entries.append(ComputedEntry.from_dict(d_))
        self.chempot_diagram = ChemicalPotentialDiagram(entries)
        if (
            bulk_entry.composition.reduced_formula not in self.chempot_diagram.domains
        ):  # pragma: no cover
            msg = (
                "Bulk entry is not stable in the chemical potential diagram."
                "Consider increasing the `bulk_stability` to make it more stable."
            )
            raise ValueError(
                msg,
            )
        chempot_limits = self.chempot_diagram.domains[
            bulk_entry.composition.reduced_formula
        ]

        if self.inc_inf_values:
            self._chempot_limits_arr = chempot_limits
        else:
            boundary_value = self.chempot_diagram.default_min_limit
            self._chempot_limits_arr = chempot_limits[
                ~np.any(chempot_limits == boundary_value, axis=1)
            ]

        self.dft_energies = {
            el: self.phase_diagram.get_hull_energy_per_atom(Composition(str(el)))
            for el in self.phase_diagram.elements
        }

    @classmethod
    def with_atomic_entries(
        cls,
        defect_entries: list[DefectEntry],
        atomic_entries: list[ComputedEntry],
        phase_diagram: PhaseDiagram,
        vbm: float,
        bulk_entry: ComputedEntry | None = None,
        **kwargs,
    ) -> FormationEnergyDiagram:
        """Create a FormationEnergyDiagram object using an existing phase diagram.

        Since the Formation energy usually looks like:

        E[Defect] - (E[Bulk] + ∑ E[Atom_i] + ∑ Chempot[Atom_i])

        The most convenient, and likely most accurate way to obtain the chemical potentials
        is to calculate the defect supercells and the atomic phases with the same level of theory.
        As long as the atomic phase energies are computed using the same settings as
        the defect supercell calculations, the method used to determine the enthalpy of
        formation of the different competing phases is not important.
        Then use the an experimentally corrected ``PhaseDiagram`` object (like the ones you can
        obtain from the Materials Project) to calculate the enthalpy of formation.

        Args:
            defect_entries:
                The list of defect entries for the different charge states.
                The finite-size correction should already be applied to these.
            atomic_entries:
                The list of entries used to construct the phase diagram and chemical
                potential diagram. They will be used to determine the stability region
                of the bulk crystal.
            phase_diagram:
                A separately computed phase diagram.
            vbm:
                The VBM of the bulk crystal.
            bulk_entry:
                The bulk computed entry to get the total energy of the bulk supercell.
            band_gap:
                The band gap of the bulk crystal. Passed directly to the \
                FormationEnergyDiagram constructor.
            inc_inf_values:
                If False these boundary points at infinity are ignored when we look at
                the chemical potential limits.
            **kwargs:
                Additional keyword arguments for the FormationEnergyDiagram class.

        Returns:
            FormationEnergyDiagram:
                The FormationEnergyDiagram object.
        """
        adjusted_entries = _get_adjusted_pd_entries(
            phase_diagram=phase_diagram,
            atomic_entries=atomic_entries,
        )
        return cls(
            bulk_entry=bulk_entry,
            defect_entries=defect_entries,
            pd_entries=adjusted_entries,
            vbm=vbm,
            **kwargs,
        )

    @classmethod
    def with_directories(
        cls,
        directory_map: dict[str, str | Path],
        defect: Defect,
        pd_entries: list[ComputedEntry],
        dielectric: float | NDArray,
        vbm: float | None = None,
        **kwargs,
    ) -> FormationEnergyDiagram:
        """Create a FormationEnergyDiagram from VASP directories.

        Args:
            directory_map: A dictionary mapping the defect name to the directory containing the
                VASP calculation.
            defect: The defect used to create the defect entries.
            pd_entries: The list of entries used to construct the phase diagram and chemical
                potential diagram. They will be used to determine the stability region
                of the bulk crystal.
            dielectric: The dielectric constant of the bulk crystal.
            vbm: The VBM of the bulk crystal.
            **kwargs: Additional keyword arguments for the constructor.
        """

        def _read_dir(directory: str | Path) -> tuple[ComputedEntry, Locpot]:
            directory = Path(directory)
            vr = Vasprun(get_zfile(Path(directory), "vasprun.xml"))
            ent = vr.get_computed_entry()
            locpot = Locpot.from_file(get_zfile(directory, "LOCPOT"))
            return ent, locpot

        if "bulk" not in directory_map:
            msg = "The bulk directory must be provided."
            raise ValueError(msg)
        bulk_entry, bulk_locpot = _read_dir(directory_map["bulk"])

        def_entries = []
        for qq, q_dir in directory_map.items():
            if qq == "bulk":
                continue
            q_entry, q_locpot = _read_dir(q_dir)
            q_d_entry = DefectEntry(
                defect=defect,
                charge_state=int(qq),
                sc_entry=q_entry,
            )

            q_d_entry.get_freysoldt_correction(
                defect_locpot=q_locpot,
                bulk_locpot=bulk_locpot,
                dielectric=dielectric,
            )
            def_entries.append(q_d_entry)
        if vbm is None:
            vr = Vasprun(get_zfile(Path(directory_map["bulk"]), "vasprun.xml"))
            vbm = vr.get_band_structure().get_vbm()["energy"]

        return cls(
            bulk_entry=bulk_entry,
            defect_entries=def_entries,
            pd_entries=pd_entries,
            vbm=vbm,
            **kwargs,
        )

    def _parse_chempots(self, chempots: dict) -> dict:
        """Parse the chemical potentials.

        Make sure that the chemical potential is represented as a dictionary.
            { Element: float }

        Args:
            chempots:
                A dictionary or list of chemical potentials.
                If a list, use the element order from self.chempot_diagram.elements.

        Returns:
            dict:
                A dictionary of chemical potentials.
        """
        if not isinstance(chempots, dict):
            chempots = {
                el: chempots[i] for i, el in enumerate(self.chempot_diagram.elements)
            }
        return chempots

    def _vbm_formation_energy(self, defect_entry: DefectEntry, chempots: dict) -> float:
        """Compute the formation energy at the VBM.

        Compute the formation energy at the VBM (essentially the y-intercept)
        for a given defect entry and set of chemical potentials.
        If the `bulk_entry` attribute is set, this will be used
        difference will be computed using that value.

        Args:
            defect_entry:
                The defect entry for which the formation energy is computed.
            chempots:
                A dictionary of chemical potentials for each element.

        Returns:
            float:
                The formation energy at the VBM.
        """
        chempots = self._parse_chempots(chempots)
        en_change = sum(
            [
                (self.dft_energies[Element(el)] + chempots[Element(el)]) * fac
                for el, fac in defect_entry.defect.element_changes.items()
            ],
        )

        try:
            ediff = defect_entry.get_ediff()
        except RuntimeError:
            ediff = defect_entry.corrected_energy - self.bulk_entry.energy

        return ediff - en_change + self.vbm * defect_entry.charge_state

    @property
    def chempot_limits(self) -> list[dict[Element, float]]:
        """Return the chemical potential limits in dictionary format."""
        return [
            dict(zip(self.chempot_diagram.elements, vertex))
            for vertex in self._chempot_limits_arr
        ]

    @property
    def competing_phases(self) -> list[dict[str, ComputedEntry]]:
        """Return the competing phases."""
        bulk_formula = self.bulk_entry.composition.reduced_formula
        cd = self.chempot_diagram
        res = []
        for pt in self._chempot_limits_arr:
            competing_phases = {}
            for hp_ent, hp in zip(cd._hyperplane_entries, cd._hyperplanes):
                if hp_ent.composition.reduced_formula == bulk_formula:
                    continue
                if _is_on_hyperplane(pt, hp):
                    competing_phases[hp_ent.composition.reduced_formula] = hp_ent
            res.append(competing_phases)
        return res

    @property
    def bulk_formula(self) -> str:
        """Get the bulk formula."""
        return self.defect_entries[0].defect.structure.composition.reduced_formula

    @property
    def defect(self) -> Defect:
        """Get the defect that this FormationEnergyDiagram represents."""
        return self.defect_entries[0].defect

    @property
    def defect_chemsys(self) -> str:
        """Get the chemical system of the defect."""
        return self.defect_entries[0].defect_chemsys

    def _get_lines(self, chempots: dict) -> list[tuple[float, float]]:
        """Get the lines for the formation energy diagram.

        Args:
            chempots:
                A dictionary of the chemical potentials (referenced to the elements)
                representations a vertex of the stability region of the chemical
                potential diagram.

        Returns:
            list[tuple[float, float]]:
                List of the slope and intercept of the lines for the formation
                energy diagram.
        """
        chempots = self._parse_chempots(chempots)
        lines = []
        for def_ent in self.defect_entries:
            b = self._vbm_formation_energy(def_ent, chempots)
            m = float(def_ent.charge_state)
            lines.append((m, b))
        return lines

    def get_transitions(
        self,
        chempots: dict,
        x_min: float = 0,
        x_max: float | None = None,
    ) -> list[tuple[float, float]]:
        """Get the transition levels for the formation energy diagram.

        Get all of the kinks in the formation energy diagram.
        The points at the VBM and CBM are given by the first and last
        point respectively.

        Args:
            chempots:
                A dictionary of the chemical potentials (referenced to the elements)
                representations a vertex of the stability region of the chemical
                potential diagram.
            x_min:
                The minimum x value (Energy) for the transition levels.
            x_max:
                The maximum x value (Energy) for the transition levels. If None, the band gap is used.

        Returns:
            Transition levels and the formation energy at each transition level.
            The first and last points are the intercepts with the
            VBM and CBM respectively.
        """
        chempots = self._parse_chempots(chempots)
        if x_max is None:  # pragma: no cover
            x_max = self.band_gap

        lines = self._get_lines(chempots)
        lines = get_lower_envelope(lines)
        return get_transitions(lines, x_min, x_max)

    def get_formation_energy(self, fermi_level: float, chempot_dict: dict) -> float:
        """Get the formation energy at a given Fermi level.

        Linearly interpolate between the transition levels.

        Args:
            fermi_level:
                The Fermi level at which the formation energy is computed.
            chempot_dict:
                A dictionary of the chemical potentials (referenced to the elemental energies).

        Returns:
            The formation energy at the given Fermi level.
        """
        transitions = np.array(
            self.get_transitions(chempot_dict, x_min=-100, x_max=100),
        )
        # linearly interpolate between the set of points
        return np.interp(fermi_level, transitions[:, 0], transitions[:, 1])

    def get_concentration(
        self,
        fermi_level: float,
        chempots: dict,
        temperature: float,
    ) -> float:
        """Get equilibrium defect concentration assuming the dilute limit.

        Args:
            fermi_level: fermi level with respect to the VBM
            chempots: Chemical potentials
            temperature: in Kelvin
        """
        chempots = self._parse_chempots(chempots=chempots)
        fe = self.get_formation_energy(fermi_level, chempots)
        return self.defect_entries[0].defect.multiplicity * fermi_dirac(
            energy=fe,
            temperature=temperature,
        )

    def as_dataframe(self) -> DataFrame:
        """Return the formation energy diagram as a pandas dataframe."""
        from pandas import DataFrame

        defect_entries = self.defect_entries
        l_ = (x.get_summary_dict() for x in defect_entries)
        return DataFrame(l_)

    def get_chempots(self, rich_element: Element | str, en_tol: float = 0.01) -> dict:
        """Get the chemical potential for a desired growth condition.

        Choose an element to be rich in, require the chemical potential of that element
        to be near the MAX energy among the points (MAX_EN - en_tol, MAX_EN), then sort
        the remaining elements by:
            1. Are they in the bulk structure:
                elements in the bulk structure are prioritized.
            2. how similar they in electron affinity to the rich element:
                dis-similar elements are prioritized.

        .. note::
            Priority 2 is pretty arbitrary, but it is a good starting point.


        Based on that priority list, take the minimum chemical potential extrema.

        Args:
            rich_element: The element that are rich in the growth condition.
            en_tol: Energy tolerance for the chemical potential of the rich element.

        Returns:
            A dictionary of the chemical potentials for the growth condition.
        """
        rich_element = Element(rich_element)
        max_val = max(self.chempot_limits, key=lambda x: x[rich_element])[rich_element]
        rich_conditions = list(
            filter(
                lambda cp: abs(cp[rich_element] - max_val) < en_tol,
                self.chempot_limits,
            ),
        )
        if len(rich_conditions) == 0:  # pragma: no cover
            msg = f"Cannot find a chemical potential condition with {rich_element} near zero."
            raise ValueError(
                msg,
            )
        # defect = self.defect_entries[0].defect
        in_bulk = self.defect_entries[0].sc_entry.composition.elements
        # make sure they are of type Element
        in_bulk = [Element(x.symbol) for x in in_bulk]
        not_in_bulk = list(set(self.chempot_limits[0].keys()) - set(in_bulk))
        in_bulk = list(filter(lambda x: x != rich_element, in_bulk))

        def el_sorter(element: Element) -> float:
            return -abs(element.electron_affinity - rich_element.electron_affinity)

        el_list = sorted(in_bulk, key=el_sorter) + sorted(not_in_bulk, key=el_sorter)

        def chempot_sorter(chempot_dict: dict) -> tuple[float, ...]:
            return tuple(chempot_dict[el] for el in el_list)

        return min(rich_conditions, key=chempot_sorter)

    def __repr__(self) -> str:
        """Representation."""
        defect_entry_summary = [
            f"\t{dent.defect.name} {dent.charge_state} {dent.corrected_energy}"
            for dent in self.defect_entries
        ]

        txt = (
            f"{self.__class__.__name__} for {self.defect.name}",
            "Defect Entries:",
            "\n".join(defect_entry_summary),
        )
        return "\n".join(txt)


@dataclass
class MultiFormationEnergyDiagram(MSONable):
    """Container for multiple formation energy diagrams."""

    formation_energy_diagrams: list[FormationEnergyDiagram]

    def __post_init__(self) -> None:
        """Set some attributes after initialization."""
        self.band_gap = self.formation_energy_diagrams[0].band_gap
        self.vbm = self.formation_energy_diagrams[0].vbm

    @classmethod
    def with_atomic_entries(
        cls,
        bulk_entry: ComputedEntry,
        defect_entries: list[DefectEntry],
        atomic_entries: list[ComputedEntry],
        phase_diagram: PhaseDiagram,
        vbm: float,
        **kwargs,
    ) -> MultiFormationEnergyDiagram:
        """Initialize using atomic entries.

        Initializes by grouping defect types, and creating a list of single
        FormationEnergyDiagram using the with_atomic_entries method (see above)

        Args:
            bulk_entry: bulk entry
            defect_entries: list of defect entries
            atomic_entries: list of atomic entries
            phase_diagram: phase diagram
            vbm: valence band maximum for the bulk phase
            **kwargs: additional kwargs for FormationEnergyDiagram
        """
        single_form_en_diagrams = []
        for _, defect_group in group_defect_entries(defect_entries=defect_entries):
            _fd = FormationEnergyDiagram.with_atomic_entries(
                bulk_entry=bulk_entry,
                defect_entries=defect_group,
                atomic_entries=atomic_entries,
                phase_diagram=phase_diagram,
                vbm=vbm,
                **kwargs,
            )
            single_form_en_diagrams.append(_fd)

        return cls(formation_energy_diagrams=single_form_en_diagrams)

    def solve_for_fermi_level(
        self,
        chempots: dict,
        temperature: float,
        dos: Dos,
    ) -> float:
        """Solves for the equilibrium fermi level at a given chempot, temperature, density of states.

        Args:
            chempots: dictionary of chemical potentials to use
            temperature: temperature at which to evaluate.
            dos: Density of states object. Must contain a structure attribute. If band_gap attribute
                is set, then dos band edges be shifted to match it.

        Returns:
            Equilibrium fermi level with respect to the valence band edge.
        """
        fdos = FermiDos(dos, bandgap=self.band_gap)
        bulk_factor = self.formation_energy_diagrams[
            0
        ].defect.structure.composition.get_reduced_formula_and_factor()[1]
        fdos_factor = fdos.structure.composition.get_reduced_formula_and_factor()[1]
        fdos_multiplicity = fdos_factor / bulk_factor
        fdos_cbm, fdos_vbm = fdos.get_cbm_vbm()

        def _get_chg(fd: FormationEnergyDiagram, ef: float) -> float:
            lines = fd._get_lines(chempots=chempots)
            return sum(
                fd.defect.multiplicity
                * charge
                * fermi_dirac(vbm_fe + charge * ef, temperature)
                for charge, vbm_fe in lines
            )

        def _get_total_q(ef: float) -> float:
            qd_tot = sum(
                _get_chg(fd=fd, ef=ef) for fd in self.formation_energy_diagrams
            )
            qd_tot += fdos_multiplicity * fdos.get_doping(
                fermi_level=ef + fdos_vbm,
                temperature=temperature,
            )
            return qd_tot

        return bisect(_get_total_q, -1.0, fdos_cbm - fdos_vbm + 1.0)


def group_defect_entries(
    defect_entries: list[DefectEntry],
    sm: StructureMatcher = None,
) -> Generator[tuple[str, list[DefectEntry]], None, None]:
    """Group defect entries by their representation.

    First by name then by structure.

    Args:
        defect_entries: list of defect entries
        sm: StructureMatcher to use for grouping

    Returns:
        Generator of (name, list of defect entries) tuples
    """
    if sm is None:
        sm = StructureMatcher(comparator=ElementComparator())

    def _get_structure(entry: DefectEntry) -> Structure:
        return entry.defect.defect_structure

    def _get_name(entry: DefectEntry) -> str:
        return entry.defect.name

    def _get_hash_no_structure(entry: DefectEntry) -> tuple[str, str]:
        return entry.defect.bulk_formula, entry.defect.name

    if all(isinstance(entry.defect, Defect) for entry in defect_entries):
        ent_groups = group_docs(
            defect_entries,
            sm=sm,
            get_structure=_get_structure,
            get_hash=_get_name,
        )
        yield from ent_groups
    elif all(isinstance(entry.defect, NamedDefect) for entry in defect_entries):
        l_ = sorted(defect_entries, key=_get_hash_no_structure)
        for _, g_entries_no_struct in groupby(l_, key=_get_hash_no_structure):
            similar_ents = list(g_entries_no_struct)
            yield similar_ents[0].defect.name, similar_ents


def group_formation_energy_diagrams(
    feds: Sequence[FormationEnergyDiagram],
    sm: StructureMatcher = None,
) -> Generator[tuple[str | None, FormationEnergyDiagram], None, None]:
    """Group formation energy diagrams by their representation.

    First by name then by structure.  Note, this function assumes that the defects
    are for the same host structure.

    Args:
        feds: list of formation energy diagrams
        sm: StructureMatcher to use for grouping

    Returns:
        Generator of (name, combined formation energy diagram) tuples.
    """
    if sm is None:
        sm = StructureMatcher(comparator=ElementComparator())

    def _get_structure(fed: FormationEnergyDiagram) -> Structure:
        return fed.defect.defect_structure

    def _get_name(fed: FormationEnergyDiagram) -> str:
        return fed.defect.name

    fed_groups = group_docs(
        feds,
        sm=sm,
        get_structure=_get_structure,
        get_hash=_get_name,
    )
    for g_name, f_group in fed_groups:
        fed = f_group[0]
        fed_d = fed.as_dict()
        dents = [dfed.defect_entries for dfed in f_group]
        fed_d["defect_entries"] = list(chain.from_iterable(dents))
        yield g_name, FormationEnergyDiagram.from_dict(fed_d)


def ensure_stable_bulk(
    pd: PhaseDiagram,
    entry: ComputedEntry,
    threshold: float = 0.001,
) -> PhaseDiagram:
    """Added entry to phase diagram and ensure that it is stable.

    Create a fake entry in the phase diagram with the same id as the supplied ``entry``
    but with energy just below the convex hull and return the updated phase diagram.

    Note: This is done regardless of whether the entry is stable or not,
    so we are effectively only using the energy from the phase diagram and ignoring
    the energy of supplied entry.

    Args:
        pd:
            Phase diagram.
        entry:
            entry to be added
        threshold:
            If the bulk energy is above the convex hull, lower it to this value below
            the convex hull.

    Returns:
        PhaseDiagram:
            Modified Phase diagram.
    """
    stable_entry = ComputedEntry(
        entry.composition,
        pd.get_hull_energy(entry.composition) - threshold,
    )
    return PhaseDiagram([*pd.all_entries, stable_entry])


def get_sc_locpot(
    uc_locpot: Locpot,
    defect_struct: Structure,
    grid_out: tuple,
    up_sample: int = 2,
    sm: StructureMatcher = None,
) -> Locpot:
    """Transform a unit cell locpot to be like a supercell locpot.

    This is useful in situations where the supercell bulk locpot is not available.
    In these cases, we will have a bulk supercell structure that is most closely
    related to the defect cell.

    Args:
        uc_locpot: Locpot object for the unit cell.
        defect_struct: Defect structure to use for the transformation.
        grid_out: grid dimensions for the supercell locpot
        up_sample: upsample factor for the supercell locpot
        sm: StructureMatcher to use for finding the transformation for UC to SC.

    Returns:
        Locpot: Locpot object for the unit cell transformed to be like the supercell.
    """
    sc_mat = get_closest_sc_mat(uc_locpot.structure, sc_struct=defect_struct, sm=sm)
    bulk_sc = uc_locpot.structure * sc_mat
    return get_volumetric_like_sc(
        uc_locpot,
        bulk_sc,
        grid_out=grid_out,
        up_sample=up_sample,
        sm=sm,
        normalization=None,
    )


def get_transitions(
    lines: list[tuple[float, float]],
    x_min: float,
    x_max: float,
) -> list[tuple[float, float]]:
    """Get the "transition" points in a list of lines.

    Given a list of lines represented as (m, b) pairs sorted in order of decreasing m.
    A "transition" point is a point where adjacent lines in the list intersect.
    i.e. intersection points (x_i, y_i) where line i intersects line i+1

    Args:
        lines: (m, b) format for each line
        x_min: minimum x value
        x_max: maximum x value

    Returns:
        List[List[float]]:
            List of intersection points, including the boundary points at
            x_min and x_max.
    """
    # make sure the lines are sorted by decreasing slope
    lines = sorted(lines, key=lambda x: x[0], reverse=True)
    transitions = [(x_min, lines[0][0] * x_min + lines[0][1])]
    for i, (m1, b1) in enumerate(lines[:-1]):
        m2, b2 = lines[i + 1]
        if m1 == m2:
            msg = "The slopes (charge states) of the set of lines should be distinct."
            raise ValueError(
                msg,
            )  # pragma: no cover
        nx, ny = ((b2 - b1) / (m1 - m2), (m1 * b2 - m2 * b1) / (m1 - m2))
        if nx < x_min:
            transitions = [(x_min, m2 * x_min + b2)]
        elif nx > x_max:
            transitions.append((x_max, m1 * x_max + b1))
            break
        else:
            transitions.append((nx, ny))
    else:
        transitions.append((x_max, lines[-1][0] * x_max + lines[-1][1]))
    return transitions


def get_lower_envelope(lines: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Get the lower envelope of the formation energy.

    Based on the fact that the lower envelope of the lines is
    given by the upper convex hull of the points (m, -b) as shown in:
    https://www.cs.umd.edu/class/spring2020/cmsc754/Lects/lect06-duality.pdf
    Note: The lines are returned with decreasing slope.

    Args:
        lines: (m, b) format for each line

    Returns:
        List[List[float]]:
            List lines that make up the lower envelope.
    """

    def _hash_float(x: float) -> float:
        return round(x, 10)

    lines_dd: dict = collections.defaultdict(lambda: float("inf"))
    for m, b in lines:
        lines_dd[_hash_float(m)] = min(lines_dd[_hash_float(m)], b)
    lines = list(lines_dd.items())

    if len(lines) < 1:  # pragma: no cover
        msg = "Need at least one line to get lower envelope."
        raise ValueError(msg)
    if len(lines) == 1:
        return lines
    if len(lines) == 2:
        return sorted(lines)

    dual_points = [(m, -b) for m, b in lines]
    upper_hull = get_upper_hull(dual_points)
    return [(m, -b) for m, b in upper_hull]


def get_upper_hull(points: ArrayLike) -> list[ArrayLike]:
    """Get the upper hull of a set of points in 2D.

    Args:
        points:
            List of points in 2D.

    Returns:
        List[(float, float)]:
            Vertices in the upper hull given from right to left.

    """
    hull = ConvexHull(points)
    vertices = hull.vertices

    # the vertices are returned in counter-clockwise order
    # so we just need to loop over the ring and get the portion
    # between the rightmost and leftmost points
    right_most_idx = max(vertices, key=lambda x: points[x][0])
    left_most_idx = min(vertices, key=lambda x: points[x][0])
    seen_right_most = False
    upper_hull = []

    # loop over the vertices twice
    for i in vertices.tolist() + vertices.tolist():
        if i == right_most_idx:
            seen_right_most = True
        if seen_right_most:
            xi, yi = points[i]
            upper_hull.append((xi, yi))
        if seen_right_most and i == left_most_idx:
            break
    return upper_hull


def _get_adjusted_pd_entries(
    phase_diagram: PhaseDiagram, atomic_entries: Sequence[ComputedEntry]
) -> list[ComputedEntry]:
    """Get the adjusted entries for the phase diagram.

    Combine the terminal energies from ``atomic_entries`` with the enthalpies of formation
    for the provided ``phase_diagram``.  To create the entries for a new phase diagram.

    Args:
        phase_diagram: Phase diagram where the enthalpies of formation are taken from.
        atomic_entries: Entries for the terminal energies.

    Returns:
        List[ComputedEntry]: Entries for the new phase diagram.
    """

    def get_interp_en(entry: ComputedEntry) -> float:
        """Get the interpolated energy of an entry."""
        e_dict = {}
        for e in atomic_entries:
            if len(e.composition.elements) != 1:  # pragma: no cover
                msg = "Only single-element entries should be provided."
                raise ValueError(
                    msg,
                )
            e_dict[e.composition.elements[0]] = e.energy_per_atom

        return sum(
            entry.composition[el] * e_dict[el] for el in entry.composition.elements
        )

    adjusted_entries = []

    for entry in phase_diagram.stable_entries:
        d_ = {
            "energy": get_interp_en(entry) + phase_diagram.get_form_energy(entry),
            "composition": entry.composition,
            "entry_id": entry.entry_id,
            "correction": 0,
        }
        adjusted_entries.append(ComputedEntry.from_dict(d_))

    return adjusted_entries


def fermi_dirac(energy: float, temperature: float) -> float:
    """Get value of fermi dirac distribution.

    Gets the defects equilibrium concentration (up to the multiplicity factor)
    at a particular fermi level, chemical potential, and temperature (in Kelvin),
    assuming dilute limit thermodynamics (non-interacting defects) using FD statistics.

    Args:
        energy: Energy of the defect with respect to the VBM.
        temperature: Temperature in Kelvin.
    """
    return 1.0 / (1.0 + np.exp((energy) / (boltzman_eV_K * temperature)))


@deprecated(
    message="Plotting functions will be moved to the the plotting module. "
    "To integrate better with MP website, we will use the Plotly library for plotting."
)
def plot_formation_energy_diagrams(
    formation_energy_diagrams: FormationEnergyDiagram
    | list[FormationEnergyDiagram]
    | MultiFormationEnergyDiagram,
    rich_element: Element | None = None,
    chempots: dict | None = None,
    alignment: float = 0.0,
    xlim: list | None = None,
    ylim: list | None = None,
    only_lower_envelope: bool = True,
    show: bool = True,
    save: bool | str = False,
    colors: list | None = None,
    legend_prefix: str | None = None,
    transition_marker: str = "*",
    transition_markersize: int = 16,
    linestyle: str = "-",
    linewidth: int = 4,
    envelope_alpha: float = 0.8,
    line_alpha: float = 0.5,
    band_edge_color: str = "k",
    filterfunction: Callable | None = None,
    legend_loc: str = "lower center",
    show_legend: bool = True,
    axis: Axes = None,
) -> Axes:
    """Plot the formation energy diagram.

    Args:
        formation_energy_diagrams: Which formation energy lines to plot.
        rich_element: The abundant element used to set the limit of the chemical potential.
        chempots: Chemical potentials at which to plot the formation energy lines
            Should be bounded by the chempot_limits property
        alignment: shift the energy axis by this amount. For example, giving bandgap/2
            will visually shift the 0 reference from the VBM to the middle of the band gap.
        xlim: Limits (low, high) to use for the x-axis. Default is to use 0eV for the
            VBM up to the band gap, plus a buffer of 0.2eV on each side
        ylim: Limits (low, high) to use for y-axis. Default is to use the minimum and
            maximum formation energy value of all defects, plus a buffer of 0.1eV
        only_lower_envelope: Whether to only plot the lower envolope (concave hull). If
            False, then the lower envolope will be highlighted, but all lines will be
            plotted.
        show: Whether to show the plot.
        save: Whether to save the plot. A string can be provided to save to a specific
            file. If True, will be saved to current working directory under the name,
            formation_energy_diagram.png
        colors: Manually select the colors to use. Must have length >= to number of
            FormationEnergyDiagrams to plot.
        legend_prefix: Prefix for all legend labels
        transition_marker: Marker style for the charge transitions
        transition_markersize: Size for charge transition markers
        linestyle: Matplotlib line style
        linewidth: Linewidth for the envelope and lines (if shown)
        envelope_alpha: Alpha for the envelope
        line_alpha: Alpha for the lines (if the are shown)
        band_edge_color: Color for VBM/CBM vertical lines
        filterfunction: A callable that filters formation energy diagram objects to clean up the plot.
        legend_loc: Location of the legend, default is "lower center".
        show_legend: Whether to show the legend.
        axis: Axis to plot on. If None, a new axis will be created.

    Returns:
        Axis subplot
    """
    if isinstance(formation_energy_diagrams, MultiFormationEnergyDiagram):
        formation_energy_diagrams = formation_energy_diagrams.formation_energy_diagrams
    elif isinstance(formation_energy_diagrams, FormationEnergyDiagram):
        formation_energy_diagrams = [formation_energy_diagrams]

    filterfunction = filterfunction if filterfunction else lambda _x: True
    formation_energy_diagrams = list(filter(filterfunction, formation_energy_diagrams))

    band_gap = formation_energy_diagrams[0].band_gap
    if not xlim and band_gap is None:
        msg = "Must specify xlim or set band_gap attribute"
        raise ValueError(msg)

    if axis is None:
        _, axis = plt.subplots()
    axis.axvline(band_gap, color=band_edge_color, linestyle="--", linewidth=1)
    axis.axvline(0, color=band_edge_color, linestyle="--", linewidth=1)
    if not xlim:
        xmin, xmax = (
            np.subtract(-0.2, alignment),
            np.subtract(band_gap + 0.2, alignment),
        )
    else:
        xmin, xmax = xlim
    ymin, ymax = np.inf, -np.inf
    legends_txt: list = []
    artists: list = []
    fontwidth = 12
    ax_fontsize = 1.3
    lg_fontsize = 10

    named_feds = []
    for name_, fed_ in group_formation_energy_diagrams(formation_energy_diagrams):
        named_feds.append((name_, fed_))

    color_line_gen = _get_line_color_and_style(colors, linestyle)
    for _fid, (fed_name, single_fed) in enumerate(named_feds):
        cur_color, cur_style = next(color_line_gen)
        chempots_ = (
            chempots
            if chempots
            else single_fed.get_chempots(rich_element=Element(rich_element))
        )
        lines = single_fed._get_lines(chempots=chempots_)
        lowerlines = get_lower_envelope(lines)
        trans = np.array(
            get_transitions(
                lowerlines,
                np.add(xmin, alignment),
                np.add(xmax, alignment),
            ),
        )
        trans_y = trans[:, 1]
        ymin = min(ymin, *trans_y)
        ymax = max(ymax, *trans_y)

        dfct: Defect = single_fed.defect_entries[0].defect
        latexname = dfct.latex_name
        if legend_prefix is not None:
            latexname = f"{legend_prefix} {latexname}"

        if ":" in fed_name:
            latexname += f" ({fed_name.split(':')[1]})"

        (_l,) = axis.plot(
            np.subtract(trans[:, 0], alignment),
            trans_y,
            color=cur_color,
            ls=cur_style,
            lw=linewidth,
            alpha=envelope_alpha,
            label=latexname,
            marker=transition_marker,
            markersize=transition_markersize,
        )
        if not only_lower_envelope:
            cur_color = _l.get_color()
            for line in lines:
                x = np.linspace(xmin, xmax)
                y = line[0] * x + line[1]
                axis.plot(
                    np.subtract(x, alignment),
                    y,
                    color=cur_color,
                    alpha=line_alpha,
                )

    axis.set_xlim(xmin, xmax)
    axis.set_ylim(ylim[0] if ylim else ymin - 0.1, ylim[1] if ylim else ymax + 0.1)
    axis.set_xlabel("Fermi energy (eV)", size=ax_fontsize * fontwidth)
    axis.set_ylabel("Defect Formation\nEnergy (eV)", size=ax_fontsize * fontwidth)
    axis.minorticks_on()
    axis.tick_params(
        which="major",
        length=8,
        width=2,
        direction="in",
        top=True,
        right=True,
        labelsize=fontwidth * ax_fontsize,
    )
    axis.tick_params(
        which="minor",
        length=2,
        width=2,
        direction="in",
        top=True,
        right=True,
        labelsize=fontwidth * ax_fontsize,
    )
    for _ax in axis.spines.values():
        _ax.set_linewidth(1.5)

    axis.axvline(0, ls="--", color="k", lw=2, alpha=0.2)
    axis.axvline(
        np.subtract(0, alignment),
        ls="--",
        color=band_edge_color,
        lw=2,
        alpha=0.8,
    )
    if band_gap:
        axis.axvline(
            np.subtract(band_gap, alignment),
            ls="--",
            color=band_edge_color,
            lw=2,
            alpha=0.8,
        )

    if show_legend:
        lg = axis.get_legend()
        if lg:
            handle, leg = lg.legend_handles, [txt._text for txt in lg.texts]
        else:
            handle, leg = [], []

        axis.legend(
            handles=artists + handle,
            labels=legends_txt + leg,
            fontsize=lg_fontsize * ax_fontsize,
            ncol=3,
            loc=legend_loc,
        )

    if save:
        save = save if isinstance(save, str) else "formation_energy_diagram.png"
        plt.savefig(save)
    if show:
        plt.show()

    return axis


def _get_line_color_and_style(
    colors: Sequence | None = None, styles: Sequence | None = None
) -> Generator[tuple[str, str], None, None]:
    """Get a generator for colors and styles.

    Create an iterator that will cycle through the colors and styles.

    Args:
        colors: List of colors to use, if None, use the default matplotlib colors.
        styles: List of styles to use, if None, will use ["-", "--", "-.", ":"]

    Returns:
        Generator of (color, style) tuples
    """
    if colors is None:
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    if styles is None:
        styles = ["-", "--", "-.", ":"]
    else:
        styles = [styles] if isinstance(styles, str) else styles

    for style in styles:
        for color in colors:
            yield color, style


def _is_on_hyperplane(pt: np.array, hp: np.array, tol: float = 1e-8) -> bool:
    """Check if a point lies on a hyperplane.

    Args:
        pt: point to check
        hp: hyperplane ((a, b, c, d) such that ax + by + cz + d = 0)
        tol: tolerance for checking if the point lies on the hyperplane

    Returns:
        bool: True if the point lies on the hyperplane
    """
    return abs(np.dot(pt, hp[:-1]) + hp[-1]) < tol

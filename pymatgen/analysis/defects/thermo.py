"""Classes and methods related to thermodynamics and energy."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from itertools import groupby
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from matplotlib import cm
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from monty.json import MSONable
from numpy.typing import ArrayLike, NDArray
from pymatgen.analysis.chempot_diagram import ChemicalPotentialDiagram
from pymatgen.analysis.phase_diagram import PhaseDiagram
from pymatgen.core import Composition, Structure
from pymatgen.electronic_structure.dos import Dos, FermiDos
from pymatgen.entries.computed_entries import ComputedEntry, ComputedStructureEntry
from pymatgen.io.vasp import Locpot, Vasprun
from scipy.constants import value as _cd
from scipy.optimize import bisect
from scipy.spatial import ConvexHull

from pymatgen.analysis.defects.core import Defect
from pymatgen.analysis.defects.corrections.freysoldt import get_freysoldt_correction
from pymatgen.analysis.defects.finder import DefectSiteFinder
from pymatgen.analysis.defects.utils import CorrectionResult, get_zfile

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
            The ComputedEntry for the bulk. If this is provided, the energy difference
            can be calculated from this automatically.
        corrections:
            A dictionary of corrections to the energy.
        corrections_metadata:
            A dictionary that acts as a generic container for storing information
            about how the corrections were calculated.  These should are only used
            for debugging and plotting purposes.
            PLEASE DO NOT USE THIS AS A CONTAINER FOR IMPORTANT DATA.

    """

    defect: Defect
    charge_state: int
    sc_entry: ComputedStructureEntry
    corrections: dict[str, float] = field(default_factory=dict)
    corrections_metadata: dict[str, Any] = field(default_factory=dict)
    sc_defect_frac_coords: tuple[float, float, float] | None = None
    bulk_entry: ComputedEntry | None = None

    def __post_init__(self) -> None:
        """Post-initialization."""
        self.charge_state = int(self.charge_state)

    def get_freysoldt_correction(
        self,
        defect_locpot: Locpot | dict,
        bulk_locpot: Locpot | dict,
        dielectric: float | NDArray,
        defect_struct: Optional[Structure] = None,
        bulk_struct: Optional[Structure] = None,
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
            kwargs:
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

        if defect_struct is None or bulk_struct is None:
            raise ValueError(
                "defect_struct and/or bulk_struct is missing either provide the structure or provide the complete locpot."
            )

        if self.sc_defect_frac_coords is None:
            finder = DefectSiteFinder()
            defect_fpos = finder.get_defect_fpos(
                defect_structure=defect_struct,
                base_structure=bulk_struct,
            )
            self.sc_defect_frac_coords = defect_fpos
        else:
            defect_fpos = self.sc_defect_frac_coords

        frey_corr = get_freysoldt_correction(
            q=self.charge_state,
            dielectric=dielectric,
            defect_locpot=defect_locpot,
            bulk_locpot=bulk_locpot,
            defect_frac_coords=defect_fpos,
            lattice=defect_struct.lattice,
            **kwargs,
        )
        self.corrections.update(
            {
                "freysoldt": frey_corr.correction_energy,
            }
        )
        self.corrections_metadata.update({"freysoldt": frey_corr.metadata.copy()})
        return frey_corr

    @property
    def corrected_energy(self) -> float:
        """The energy of the defect entry with all corrections applied."""
        return self.sc_entry.energy + self.corrections["freysoldt"]

    def get_ediff(self) -> float | None:
        """Get the energy difference between the defect and the bulk (including finite-size correction)."""
        if self.bulk_entry:
            return self.corrected_energy - self.bulk_entry.energy
        else:
            return None


@dataclass
class FormationEnergyDiagram(MSONable):
    """Formation energy.

    Attributes:
        bulk_entry:
            The bulk computed entry to get the total energy of the bulk supercell.
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
        inc_inf_values:
            If False these boundary points at infinity are ignored when we look at the
            chemical potential limits.
            The stability region is sometimes unbounded, example:
            Mn_Ga in GaN, the chemical potential of Mn is does not affect
            the stability of GaN so it can go to ``-inf``.
            A artificial value is needed to help the half-space intersection algorithm.
            This can be justified since these tend to be the substitutional elements
            which should not have very negative chemical potential.

    """

    defect_entries: List[DefectEntry]
    pd_entries: list[ComputedEntry]
    vbm: float
    band_gap: Optional[float] = None
    bulk_entry: Optional[ComputedStructureEntry] = None
    inc_inf_values: bool = False

    def __post_init__(self):
        """Post-initialization.

        - Reconstruct the phase diagram with the bulk entry
        - Make sure that the bulk entry is stable
        - create the chemical potential diagram using only the formation energies
        """
        g = group_defects(self.defect_entries)
        if next(g, True) and next(g, False):
            raise ValueError(
                "Defects are not of same type! "
                "Use MultiFormationEnergyDiagram for multiple defect types"
            )
        # if all of the `DefectEntry` objects have the same `bulk_entry` then `self.bulk_entry` is not needed
        if all(d.bulk_entry is not None for d in self.defect_entries):
            if self.bulk_entry is not None:
                raise RuntimeError(
                    "All of the `DefectEntry` objects have a `bulk_entry` attribute, it is not needed to provide `bulk_entry` to `FormationEnergyDiagram`"
                    "The energy difference E[Defect SuperCell] - E[Bulk SuperCell] can be calculated directly from `DefectEntry.get_ediff`."
                )
        else:
            if self.bulk_entry is None:
                raise RuntimeError(
                    "Not all of the `DefectEntry` objects have a `bulk_entry` attribute, you need to provide `bulk_entry` to `FormationEnergyDiagram`"
                )

        bulk_entry = self.bulk_entry or self.defect_entries[0].bulk_entry
        pd_ = PhaseDiagram(self.pd_entries)
        entries = pd_.stable_entries | {bulk_entry}
        pd_ = PhaseDiagram(entries)
        self.phase_diagram = ensure_stable_bulk(pd_, bulk_entry)
        entries = []
        for entry in self.phase_diagram.stable_entries:
            d_ = dict(
                energy=self.phase_diagram.get_form_energy(entry),
                composition=entry.composition,
                entry_id=entry.entry_id,
                correction=0.0,
            )
            entries.append(ComputedEntry.from_dict(d_))
            entries.append(ComputedEntry.from_dict(d_))

        self.chempot_diagram = ChemicalPotentialDiagram(entries)
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
    ):
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
                The band gap of the bulk crystal.
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
            phase_diagram=phase_diagram, atomic_entries=atomic_entries
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
        directory_map: Dict[str, str],
        defect: Defect,
        pd_entries: list[ComputedEntry],
        dielectric: float | NDArray,
        vbm: float | None = None,
        **kwargs,
    ):
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

        def _read_dir(directory):
            vr = Vasprun(get_zfile(Path(directory), "vasprun.xml"))
            ent = vr.get_computed_entry()
            locpot = Locpot.from_file(get_zfile(directory, "LOCPOT"))
            return ent, locpot

        if "bulk" not in directory_map:
            raise ValueError("The bulk directory must be provided.")
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
                defect_locpot=q_locpot, bulk_locpot=bulk_locpot, dielectric=dielectric
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

        Args:
            defect_entry:
                The defect entry for which the formation energy is computed.
            chem_pots:
                A dictionary of chemical potentials for each element.

        Returns:
            float:
                The formation energy at the VBM.
        """
        chempots = self._parse_chempots(chempots)
        en_change = sum(
            [
                (self.dft_energies[el] + chempots[el]) * fac
                for el, fac in defect_entry.defect.element_changes.items()
            ]
        )
        ediff = defect_entry.get_ediff()
        if ediff is not None:
            formation_en = ediff - en_change + self.vbm * defect_entry.charge_state
        else:
            formation_en = (
                defect_entry.corrected_energy
                - self.bulk_entry.energy
                - en_change
                + self.vbm * defect_entry.charge_state
            )
        return formation_en

    @property
    def chempot_limits(self):
        """Return the chemical potential limits in dictionary format."""
        res = []
        for vertex in self._chempot_limits_arr:
            res.append(dict(zip(self.chempot_diagram.elements, vertex)))
        return res

    @property
    def defect(self):
        """Get the defect that this FormationEnergyDiagram represents."""
        return self.defect_entries[0].defect

    def _get_lines(self, chempots: Dict) -> list[tuple[float, float]]:
        """Get the lines for the formation energy diagram.

        Args:
            chempot_dict:
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
        self, chempots: dict, x_min: float = 0, x_max: float = 10
    ) -> list[tuple[float, float]]:
        """Get the transition levels for the formation energy diagram.

        Get all of the kinks in the formation energy diagram.
        The points at the VBM and CBM are given by the first and last
        point respectively.

        Args:
            chempot_dict:
                A dictionary of the chemical potentials (referenced to the elements)
                representations a vertex of the stability region of the chemical
                potential diagram.

        Returns:
            Transition levels and the formation energy at each transition level.
            The first and last points are the intercepts with the
            VBM and CBM respectively.
        """
        chempots = self._parse_chempots(chempots)
        if x_max is None:
            x_max = self.band_gap

        lines = self._get_lines(chempots)
        lines = get_lower_envelope(lines)
        return get_transitions(lines, x_min, x_max)

    def get_formation_energy(self, fermi_level: float, chempot_dict: dict):
        """Get the formation energy at a given Fermi level.

        Linearly interpolate between the transition levels.

        Args:
            fermi_level:
                The Fermi level at which the formation energy is computed.

        Returns:
            The formation energy at the given Fermi level.
        """
        transitions = np.array(
            self.get_transitions(chempot_dict, x_min=-100, x_max=100)
        )
        # linearly interpolate between the set of points
        return np.interp(fermi_level, transitions[:, 0], transitions[:, 1])

    def get_concentration(
        self, fermi_level: float, chempots: dict, temperature: int | float
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
            energy=fe, temperature=temperature
        )


@dataclass
class MultiFormationEnergyDiagram(MSONable):
    """Container for multiple formation energy diagrams."""

    formation_energy_diagrams: List[FormationEnergyDiagram]

    def __post_init__(self):
        """Set some attributes after initialization."""
        self.band_gap = self.formation_energy_diagrams[0].band_gap
        self.vbm = self.formation_energy_diagrams[0].vbm
        self.chempot_limits = self.formation_energy_diagrams[0].chempot_limits
        self.chempot_diagram = self.formation_energy_diagrams[0].chempot_diagram

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
        """
        single_form_en_diagrams = []
        for _, defect_group in group_defects(defect_entries=defect_entries):
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
        self, chempots: dict, temperature: int | float, dos: Dos
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

        def _get_chg(fd: FormationEnergyDiagram, ef):
            lines = fd._get_lines(chempots=chempots)
            return sum(
                fd.defect.multiplicity
                * charge
                * fermi_dirac(vbm_fe + charge * ef, temperature)
                for charge, vbm_fe in lines
            )

        def _get_total_q(ef):
            qd_tot = sum(
                _get_chg(fd=fd, ef=ef) for fd in self.formation_energy_diagrams
            )
            qd_tot += fdos_multiplicity * fdos.get_doping(
                fermi_level=ef + fdos_vbm, temperature=temperature
            )
            return qd_tot

        return bisect(_get_total_q, -1.0, fdos_cbm - fdos_vbm + 1.0)


def group_defects(defect_entries: list[DefectEntry]):
    """Group defects by their representation.

    TODO: Fix this with `defect_structure` grouping with `StructureMatcher`.
    """
    sents = sorted(defect_entries, key=lambda x: x.defect.__repr__())
    for k, group in groupby(sents, key=lambda x: x.defect.__repr__()):
        yield k, list(group)


def ensure_stable_bulk(
    pd: PhaseDiagram,
    entry: ComputedEntry,
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

    Returns:
        PhaseDiagram:
            Modified Phase diagram.
    """
    SMALL_NUM = 1e-8
    stable_entry = ComputedEntry(
        entry.composition, pd.get_hull_energy(entry.composition) - SMALL_NUM
    )
    pd = PhaseDiagram(pd.all_entries + [stable_entry])
    return pd


def get_transitions(
    lines: list[tuple[float, float]], x_min: float, x_max: float
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
            raise ValueError(
                "The slopes (charge states) of the set of lines should be distinct."
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


def get_lower_envelope(lines):
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
    if len(lines) < 1:
        raise ValueError("Need at least one line to get lower envelope.")
    elif len(lines) == 1:
        return lines
    elif len(lines) == 2:
        return sorted(lines)

    dual_points = [(m, -b) for m, b in lines]
    upper_hull = get_upper_hull(dual_points)
    lower_envelope = [(m, -b) for m, b in upper_hull]
    return lower_envelope


def get_upper_hull(points: ArrayLike) -> List[ArrayLike]:
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


def _get_adjusted_pd_entries(phase_diagram, atomic_entries) -> list[ComputedEntry]:
    """Get the adjusted entries for the phase diagram.

    Combine the terminal energies from ``atomic_entries`` with the enthalpies of formation
    for the provided ``phase_diagram``.  To create the entries for a new phase diagram.

    Args:
        phase_diagram: Phase diagram where the enthalpies of formation are taken from.
        atomic_entries: Entries for the terminal energies.

    Returns:
        List[ComputedEntry]: Entries for the new phase diagram.
    """

    def get_interp_en(entry: ComputedEntry):
        """Get the interpolated energy of an entry."""
        e_dict = dict()
        for e in atomic_entries:
            if len(e.composition.elements) != 1:
                raise ValueError(
                    "Only single-element entries should be provided."
                )  # pragma: no cover
            e_dict[e.composition.elements[0]] = e.energy_per_atom

        return sum(
            entry.composition[el] * e_dict[el] for el in entry.composition.elements
        )

    adjusted_entries = []

    for entry in phase_diagram.stable_entries:
        d_ = dict(
            energy=get_interp_en(entry) + phase_diagram.get_form_energy(entry),
            composition=entry.composition,
            entry_id=entry.entry_id,
            correction=0,
        )
        adjusted_entries.append(ComputedEntry.from_dict(d_))

    return adjusted_entries


def fermi_dirac(energy: float, temperature: int | float) -> float:
    """Get value of fermi dirac distribution.

    Gets the defects equilibrium concentration (up to the multiplicity factor)
    at a particular fermi level, chemical potential, and temperature (in Kelvin),
    assuming dilue limit thermodynamics (non-interacting defects) using FD statistics.
    """
    return 1.0 / (1.0 + np.exp((energy) / (boltzman_eV_K * temperature)))


def plot_formation_energy_diagrams(
    formation_energy_diagrams: FormationEnergyDiagram
    | List[FormationEnergyDiagram]
    | MultiFormationEnergyDiagram,
    chempots: Dict,
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
    band_edge_color="k",
    filterfunction: Callable | None = None,
    axis=None,
):
    """Plot the formation energy diagram.

    Args:
        formation_energy_diagrams: Which formation energy lines to plot.
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
        filterfunction: A callable that filters formation energy diagram objects to clean up the plot
        axis: Previous axis to amend

    Returns:
        Axis subplot
    """
    if isinstance(formation_energy_diagrams, MultiFormationEnergyDiagram):
        formation_energy_diagrams = formation_energy_diagrams.formation_energy_diagrams
    elif isinstance(formation_energy_diagrams, FormationEnergyDiagram):
        formation_energy_diagrams = [formation_energy_diagrams]

    filterfunction = filterfunction if filterfunction else lambda x: True
    formation_energy_diagrams = list(filter(filterfunction, formation_energy_diagrams))

    band_gap = formation_energy_diagrams[0].band_gap
    if not xlim and not band_gap:
        raise ValueError("Must specify xlim or set band_gap attribute")

    if not axis:
        _, axis = plt.subplots()
    if not xlim and band_gap:
        xmin, xmax = np.subtract(-0.2, alignment), np.subtract(
            band_gap + 0.2, alignment
        )
    else:
        xmin, xmax = xlim
    ymin, ymax = 0.0, 1.0
    legends_txt = []
    artists = []
    fontwidth = 12
    ax_fontsize = 1.3
    lg_fontsize = 10

    colors = (
        colors
        if colors
        else cm.Dark2(np.linspace(0, 1, len(formation_energy_diagrams)))
        if len(formation_energy_diagrams) <= 8
        else cm.gist_rainbow(np.linspace(0, 1, len(formation_energy_diagrams)))
    )

    for fid, single_fed in enumerate(formation_energy_diagrams):
        lines = single_fed._get_lines(chempots=chempots)
        lowerlines = get_lower_envelope(lines)
        trans = get_transitions(
            lowerlines, np.add(xmin, alignment), np.add(xmax, alignment)
        )

        # plot lines
        if not only_lower_envelope:
            for line in lines:
                x = np.linspace(xmin, xmax)
                y = line[0] * x + line[1]
                axis.plot(
                    np.subtract(x, alignment), y, color=colors[fid], alpha=line_alpha
                )

        # plot connecting envelop lines
        for i, (_x, _y) in enumerate(trans[:-1]):
            x = np.linspace(_x, trans[i + 1][0])
            y = ((trans[i + 1][1] - _y) / (trans[i + 1][0] - _x)) * (x - _x) + _y
            axis.plot(
                np.subtract(x, alignment),
                y,
                color=colors[fid],
                ls=linestyle,
                lw=linewidth,
                alpha=envelope_alpha,
            )

        # Plot transitions
        for _x, _y in trans:
            ymax = max((ymax, _y))
            ymin = min((ymin, _y))
            axis.plot(
                np.subtract(_x, alignment),
                _y,
                marker=transition_marker,
                color=colors[fid],
                markersize=transition_markersize,
            )

        # get latex-like legend titles
        dfct = single_fed.defect_entries[0].defect
        flds = dfct.name.split("_")
        latexname = f"${flds[0]}_{{{flds[1]}}}$"
        if legend_prefix:
            latexname = f"{legend_prefix} {latexname}"
        legends_txt.append(latexname)
        artists.append(Line2D([0], [0], color=colors[fid], lw=4))

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
        np.subtract(0, alignment), ls="--", color=band_edge_color, lw=2, alpha=0.8
    )
    if band_gap:
        axis.axvline(
            np.subtract(band_gap, alignment),
            ls="--",
            color=band_edge_color,
            lw=2,
            alpha=0.8,
        )

    lg = axis.get_legend()
    if lg:
        handle, leg = lg.legendHandles, [txt._text for txt in lg.texts]
    else:
        handle, leg = [], []
    axis.legend(
        handles=artists + handle,
        labels=legends_txt + leg,
        fontsize=lg_fontsize * ax_fontsize,
        ncol=3,
        loc="lower center",
    )

    if save:
        save = save if isinstance(save, str) else "formation_energy_diagram.png"
        plt.savefig(save)
    if show:
        plt.show()

    return axis

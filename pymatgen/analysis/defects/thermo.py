"""Classes and methods related to thermodynamics and energy."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from itertools import groupby
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.cm as cm
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from monty.json import MSONable
from numpy.typing import ArrayLike, NDArray
from pymatgen.analysis.chempot_diagram import ChemicalPotentialDiagram
from pymatgen.analysis.phase_diagram import PhaseDiagram
from pymatgen.core import Composition
from pymatgen.electronic_structure.dos import Dos, FermiDos
from pymatgen.entries.computed_entries import ComputedEntry, ComputedStructureEntry
from pymatgen.io.vasp import Locpot, Vasprun
from scipy.constants import value as _cd
from scipy.optimize import bisect
from scipy.spatial import ConvexHull

from pymatgen.analysis.defects.core import Defect
from pymatgen.analysis.defects.corrections import get_freysoldt_correction
from pymatgen.analysis.defects.finder import DefectSiteFinder
from pymatgen.analysis.defects.utils import get_zfile

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
        corrections:
            A dictionary of corrections to the energy.
        correction_summaries:
            A dictionary that acts as a generic container for storing information
            about how the corrections were calculated.  These should are only used
            for debugging and plotting purposes.
    """

    defect: Defect
    charge_state: int
    sc_entry: ComputedStructureEntry
    sc_defect_frac_coords: Optional[ArrayLike] = None
    corrections: Optional[Dict[str, float]] = None
    corrections_summaries: Optional[Dict[str, Dict]] = None

    def __post_init__(self):
        """Post-initialization."""
        self.charge_state = int(self.charge_state)
        self.corrections: dict = {} if self.corrections is None else self.corrections
        self.corrections_summaries: dict = (
            {} if self.corrections_summaries is None else self.corrections_summaries
        )

    def get_freysoldt_correction(
        self,
        defect_locpot: Locpot,
        bulk_locpot: Locpot,
        dielectric: float | NDArray,
        **kwargs,
    ):
        """Calculate the Freysoldt correction.

        Updates the corrections dictionary with the Freysoldt correction
        and returns the planar averaged potential data for plotting.

        Args:
            defect_locpot:
                The Locpot object for the defect supercell.
            bulk_locpot:
                The Locpot object for the bulk supercell.
            dielectric:
                The dielectric tensor or constant for the bulk material.
            kwargs:
                Additional keyword arguments for the get_correction method.

        Returns:
            dict:
                The plotting data to analyze the planar averaged electrostatic potential
                in the three periodic lattice directions.


        """
        if self.sc_defect_frac_coords is None:
            finder = DefectSiteFinder()
            defect_fpos = finder.get_defect_fpos(
                defect_structure=defect_locpot.structure,
                base_structure=bulk_locpot.structure,
            )
        else:
            defect_fpos = self.sc_defect_frac_coords

        frey_corr, plot_data = get_freysoldt_correction(
            q=self.charge_state,
            dielectric=dielectric,
            defect_locpot=defect_locpot,
            bulk_locpot=bulk_locpot,
            defect_frac_coords=defect_fpos,
            **kwargs,
        )
        self.corrections.update(frey_corr)  # type: ignore
        self.corrections_summaries["freysoldt_corrections"] = plot_data.copy()
        return plot_data

    @property
    def corrected_energy(self):
        """The energy of the defect entry with all corrections applied."""
        return self.sc_entry.energy + sum(self.corrections.values())


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

    bulk_entry: ComputedStructureEntry
    defect_entries: List[DefectEntry]
    pd_entries: list[ComputedEntry]
    vbm: float
    band_gap: Optional[float] = None
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

        pd_ = PhaseDiagram(self.pd_entries)
        entries = pd_.stable_entries | {self.bulk_entry}
        pd_ = PhaseDiagram(entries)
        self.phase_diagram = ensure_stable_bulk(pd_, self.bulk_entry)

        entries = []
        for entry in self.phase_diagram.stable_entries:
            d_ = entry.as_dict()
            d_["energy"] = self.phase_diagram.get_form_energy(entry) - entry.correction
            entries.append(ComputedEntry.from_dict(d_))

        self.chempot_diagram = ChemicalPotentialDiagram(entries)
        chempot_limits = self.chempot_diagram.domains[
            self.bulk_entry.composition.reduced_formula
        ]

        if self.inc_inf_values:
            self._chempot_limits_arr = chempot_limits
        else:
            boundary_value = self.chempot_diagram.default_min_limit
            self._chempot_limits_arr = chempot_limits[
                ~np.any(chempot_limits == boundary_value, axis=1)
            ]
        self._chempot_limits_arr.dot(
            1 / self.bulk_entry.composition.reduced_composition.num_atoms
        )

        self.dft_energies = {
            el: self.phase_diagram.get_hull_energy_per_atom(Composition(str(el)))
            for el in self.phase_diagram.elements
        }

    @classmethod
    def with_atomic_entries(
        cls,
        bulk_entry: ComputedEntry,
        defect_entries: list[DefectEntry],
        atomic_entries: list[ComputedEntry],
        phase_diagram: PhaseDiagram,
        vbm: float,
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
        Then use the an exerimentally corrected ``PhaseDiagram`` object (like the ones you can
        obtain from the Materials Project) to calculate the enthalpy of formation.

        Args:
            bulk_entry:
                The bulk computed entry to get the total energy of the bulk supercell.
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
        formation_en = (
            defect_entry.corrected_energy
            - (self.bulk_entry.energy + en_change)
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
            Organized as a dictionary keyed by defect __repr__
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
            dos: Density of states object. Must contain a structure attribute.

        Returns:
            Equilibrium fermi level with respect to the valence band edge.
        """
        fdos = FermiDos(dos, bandgap=self.band_gap)
        bulk_factor = (
            self.defect.structure.composition.get_reduced_formula_and_factor()[1]
        )
        fdos_factor = fdos.structure.composition.get_reduced_formula_and_factor()[1]
        fdos_multiplicity = fdos_factor / bulk_factor
        _, fdos_vbm = fdos.get_cbm_vbm()

        def _get_chg(fd: FormationEnergyDiagram, ef):
            lines = fd._get_lines(chempots=chempots)
            return sum(
                self.defect.multiplicity
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

        return bisect(_get_total_q, -1.0, self.band_gap + 1.0)


def group_defects(defect_entries: list[DefectEntry]):
    """Group defects by their representation."""
    sents = sorted(defect_entries, key=lambda x: x.defect.__repr__())
    for k, group in groupby(sents, key=lambda x: x.defect.__repr__()):
        yield k, list(group)


def ensure_stable_bulk(
    pd: PhaseDiagram, entry: ComputedEntry, use_pd_energy: bool = True
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
    e_above_hull = pd.get_e_above_hull(entry)
    stable_entry = ComputedEntry(
        entry.composition, entry.energy - e_above_hull - SMALL_NUM
    )
    pd = PhaseDiagram([stable_entry] + pd.all_entries)
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
            energy=get_interp_en(entry)
            + phase_diagram.get_form_energy(entry)
            - entry.correction,
            composition=entry.composition,
            entry_id=entry.entry_id,
            correction=entry.correction,
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
    xlim: ArrayLike | None = None,
    ylim: ArrayLike | None = None,
    only_lower_envelope: bool = True,
    show: bool = True,
    save: bool | str = False,
    colors: List | None = None,
    legend_prefix: str | None = None,
    transition_marker: str = "*",
    transition_markersize: int = 16,
    linestyle: str = "-",
    linewidth: int = 4,
    envelope_alpha: float = 0.8,
    line_alpha: float = 0.5,
    band_edge_color="k",
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
        axis: Previous axis to ammend

    Returns:
        Axis subplot
    """
    if isinstance(formation_energy_diagrams, MultiFormationEnergyDiagram):
        formation_energy_diagrams = formation_energy_diagrams.formation_energy_diagrams
    elif isinstance(formation_energy_diagrams, FormationEnergyDiagram):
        formation_energy_diagrams = [formation_energy_diagrams]

    band_gap = formation_energy_diagrams[0].band_gap
    if not xlim and not band_gap:
        raise ValueError("Must specify xlim or set band_gap attribute")

    if not axis:
        _, axis = plt.subplots()
    xmin = xlim[0] if xlim else np.subtract(-0.2, alignment)
    xmax = xlim[1] if xlim else np.subtract(band_gap + 0.2, alignment)
    ymin, ymax = 10, 0
    legends_txt = []
    artists = []
    fontwidth = 12
    ax_fontsize = 1.3
    lg_fontsize = 10

    if not colors and len(formation_energy_diagrams) <= 8:
        colors = iter(cm.Dark2(np.linspace(0, 1, len(formation_energy_diagrams))))
    elif not colors:
        colors = iter(
            cm.gist_rainbow(np.linspace(0, 1, len(formation_energy_diagrams)))
        )

    for single_fed in formation_energy_diagrams:
        color = next(colors)
        lines = single_fed._get_lines(chempots=chempots)
        lowerlines = get_lower_envelope(lines)
        trans = get_transitions(
            lowerlines, np.add(xmin, alignment), np.add(xmax, alignment)
        )

        # plot lines
        if not only_lower_envelope:
            for ln in lines:
                x = np.linspace(xmin, xmax)
                y = ln[0] * x + ln[1]
                axis.plot(np.subtract(x, alignment), y, color=color, alpha=line_alpha)

        # plot connecting envelop lines
        for i, (_x, _y) in enumerate(trans[:-1]):
            x = np.linspace(_x, trans[i + 1][0])
            y = ((trans[i + 1][1] - _y) / (trans[i + 1][0] - _x)) * (x - _x) + _y
            axis.plot(
                np.subtract(x, alignment),
                y,
                color=color,
                ls=linestyle,
                lw=linewidth,
                alpha=envelope_alpha,
            )

        # Plot transitions
        for x, y in trans:
            ymax = max((ymax, y))
            ymin = min((ymin, y))
            axis.plot(
                np.subtract(x, alignment),
                y,
                marker=transition_marker,
                color=color,
                markersize=transition_markersize,
            )

        # get latex-like legend titles
        dfct = single_fed.defect_entries[0].defect
        flds = dfct.name.split("_")
        latexname = f"${flds[0]}_{{{flds[1]}}}$"
        if legend_prefix:
            latexname = f"{legend_prefix} {latexname}"
        legends_txt.append(latexname)
        artists.append(Line2D([0], [0], color=color, lw=4))

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
        h, l = lg.legendHandles, [l._text for l in lg.texts]
    else:
        h, l = [], []
    axis.legend(
        handles=artists + h,
        labels=legends_txt + l,
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

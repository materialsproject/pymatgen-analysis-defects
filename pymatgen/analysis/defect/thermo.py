"""Classes and methods related to thermodynamics and energy."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import numpy as np
from monty.json import MSONable
from numpy.typing import ArrayLike, NDArray
from pymatgen.analysis.chempot_diagram import ChemicalPotentialDiagram
from pymatgen.analysis.phase_diagram import PhaseDiagram
from pymatgen.core import Composition
from pymatgen.entries.computed_entries import ComputedEntry, ComputedStructureEntry
from pymatgen.io.vasp.outputs import Locpot
from scipy.spatial import ConvexHull

from pymatgen.analysis.defect.core import Defect
from pymatgen.analysis.defect.corrections import get_correction
from pymatgen.analysis.defect.finder import DefectSiteFinder

__author__ = "Jimmy-Xuan Shen, Danny Broberg, Shyam Dwaraknath"
__copyright__ = "Copyright 2022, The Materials Project"
__maintainer__ = "Jimmy-Xuan Shen"
__email__ = "jmmshn@gmail.com"

_logger = logging.getLogger(__name__)


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
        dielectric:
            The dielectric tensor or constant for the bulk material.
        sc_defect_frac_coords:
            The fractional coordinates of the defect in the supercell.
            If None, structures attributes of the locpot file will be used to automatically determine
            the defect location.
        corrections:
            A dictionary of corrections to the energy.
    """

    defect: Defect
    charge_state: int
    sc_entry: ComputedStructureEntry
    dielectric: float | NDArray
    sc_defect_frac_coords: Optional[ArrayLike] = None
    corrections: Optional[Dict[str, float]] = None

    def __post_init__(self):
        """Post-initialization."""
        self.charge_state = int(self.charge_state)
        self.corrections: dict = {} if self.corrections is None else self.corrections

    def get_freysoldt_correction(self, defect_locpot: Locpot, bulk_locpot: Locpot, **kwargs):
        """Calculate the Freysoldt correction.

        Updates the corrections dictionary with the Freysoldt correction
        and returns the planar averaged potential data for plotting.

        Args:
            defect_locpot:
                The Locpot object for the defect supercell.
            bulk_locpot:
                The Locpot object for the bulk supercell.
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

        frey_corr, plot_data = get_correction(
            q=self.charge_state,
            dielectric=self.dielectric,
            defect_locpot=defect_locpot,
            bulk_locpot=bulk_locpot,
            defect_frac_coords=defect_fpos,
            **kwargs,
        )

        self.corrections.update(frey_corr)  # type: ignore
        return plot_data

    @property
    def corrected_energy(self):
        """The energy of the defect entry with all corrections applied."""
        return self.sc_entry.energy + sum(self.corrections.values())


@dataclass
class FormationEnergyDiagram(MSONable):
    """Formation energy."""

    bulk_entry: ComputedStructureEntry
    defect_entries: List[DefectEntry]
    vbm: float
    pd_entries: list[ComputedEntry]
    band_gap: Optional[float] = None

    def __post_init__(self):
        """Post-initialization.

        - Reconstruct the phase diagram with the bulk entry
        - Make sure that the bulk entry is stable
        - create the chemical potential diagram using only the formation energies
        """
        pd_ = PhaseDiagram(self.pd_entries)
        entries = pd_.stable_entries | {self.bulk_entry}
        pd_ = PhaseDiagram(entries)
        self.phase_diagram = ensure_stable_bulk(pd_, self.bulk_entry)

        entries = []
        for entry in self.phase_diagram.stable_entries:
            d_ = entry.as_dict()
            d_["energy"] = self.phase_diagram.get_form_energy(entry)
            entries.append(ComputedEntry.from_dict(d_))

        self.chempot_diagram = ChemicalPotentialDiagram(entries)
        self.chempot_limits = self.chempot_diagram.domains[self.bulk_entry.composition.reduced_formula]
        boundary_value = self.chempot_diagram.default_min_limit
        self.chempot_limits = self.chempot_limits[~np.any(self.chempot_limits == boundary_value, axis=1)]

        self.el_change = self.defect_entries[0].defect.element_changes
        self.dft_energies = {
            el: self.phase_diagram.get_hull_energy_per_atom(Composition(str(el))) for el in self.phase_diagram.elements
        }

    def _parse_chempots(self, chempots: dict | ArrayLike) -> dict:
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
            chempots = {el: chempots[i] for i, el in enumerate(self.chempot_diagram.elements)}
        return chempots

    def _vbm_formation_energy(self, defect_entry: DefectEntry, chempots: dict | Iterable) -> float:
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
        en_change = sum([(self.dft_energies[el] + chempots[el]) * fac for el, fac in self.el_change.items()])
        formation_en = (
            defect_entry.corrected_energy - (self.bulk_entry.energy + en_change) + self.vbm * defect_entry.charge_state
        )
        return formation_en

    def get_limit_formation_energies(self, compute_lower_hull: bool = True):
        """Compute the formation energy diagram for all chemical potential limits.

        Returns:
            List containing the formation energy at each chemical potential limit.
        """
        res = []
        for vertex in self.chempot_limits:
            limit_dict = dict(zip(self.chempot_diagram.elements, vertex))
            lines = self._get_lines(limit_dict)
            if compute_lower_hull:
                lines = get_lower_envelope(lines)
            res.append(lines)
        return lines

    def _get_lines(self, chempots: dict) -> list[tuple[float, float]]:
        """Get the lines for the formation energy diagram.

        Args:
            chempot_dict:
                A dictionary of the chemical potentials (referenced to the elements) representations
                a vertex of the stability region of the chemical potential diagram.

        Returns:
            list[tuple[float, float]]:
                List of the slope and intercept of the lines for the formation energy diagram.
        """
        chempots = self._parse_chempots(chempots)
        lines = []
        for def_ent in self.defect_entries:
            b = self._vbm_formation_energy(def_ent, chempots)
            m = float(def_ent.charge_state)
            lines.append((m, b))
        return lines

    def get_transitions(
        self, chempots: dict, x_min: float = 0, x_max: float | None = None
    ) -> list[tuple[float, float]]:
        """Get the transition levels for the formation energy diagram.

        Get all of the kinks in the formation energy diagram.
        The points at the VBM and CBM are given by the first and last point respectively.

        Args:
            chempot_dict:
                A dictionary of the chemical potentials (referenced to the elements) representations
                a vertex of the stability region of the chemical potential diagram.

        Returns:
            list[tuple[float, float]]:
                Transition levels and the formation energy at each transition level.
                The first and last points are the intercepts with the VBM and CBM respectively.
        """
        chempots = self._parse_chempots(chempots)
        if x_max is None:
            x_max = getattr(self, "band_gap")
        lines = self._get_lines(chempots)
        lines = get_lower_envelope(lines)
        return get_transitions(lines, x_min, x_max)

    def _get_formation_energy(self, fermi_level: float, chempot_dict: dict):
        """Get the formation energy at a given Fermi level.

        Linearly interpolate between the transition levels.

        Args:
            fermi_level:
                The Fermi level at which the formation energy is computed.

        Returns:
            The formation energy at the given Fermi level.
        """
        transitions = np.array(self.get_transitions(chempot_dict, x_min=-100, x_max=100))
        # linearly interpolate between the set of points
        return np.interp(fermi_level, transitions[:, 0], transitions[:, 1])


def ensure_stable_bulk(pd: PhaseDiagram, bulk_entry: ComputedEntry) -> PhaseDiagram:
    """Ensure the bulk is stable.

    Args:
        pd:
            Phase diagram.
        bulk_entry:
            Bulk entry.

    Returns:
        PhaseDiagram:
            Modified Phase diagram.
    """
    SMALL_NUM = 1e-8
    e_above_hull = pd.get_e_above_hull(bulk_entry)
    if e_above_hull > 0:
        logging.warning(
            f"Bulk entry is unstable. E above hull: {e_above_hull}\n"
            f"Adding a fake entry just below the hull to get estimates of the chemical potentials."
        )
        stable_entry = ComputedEntry(bulk_entry.composition, bulk_entry.energy - e_above_hull - SMALL_NUM)
        pd = PhaseDiagram([stable_entry] + pd.all_entries)
    return pd


def get_transitions(lines: list[tuple[float, float]], x_min: float, x_max: float) -> list[tuple[float, float]]:
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
            List of intersection points, including the boundary points at x_min and x_max.
    """
    # make sure the lines are sorted by decreasing slope
    lines = sorted(lines, key=lambda x: x[0], reverse=True)
    transitions = [(x_min, lines[0][0] * x_min + lines[0][1])]
    for i, (m1, b1) in enumerate(lines[:-1]):
        m2, b2 = lines[i + 1]
        if m1 == m2:
            raise ValueError("The slopes (charge states) of the set of lines should be distinct.")
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

"""Classes and methods related to thermodynamics and energy."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from monty.json import MSONable
from numpy.typing import ArrayLike, NDArray
from pymatgen.analysis.phase_diagram import PhaseDiagram
from pymatgen.core import Element
from pymatgen.entries.computed_entries import ComputedEntry, ComputedStructureEntry
from pymatgen.io.vasp.outputs import Locpot
from scipy.spatial import ConvexHull

from pymatgen.analysis.defect.core import Defect
from pymatgen.analysis.defect.corrections import FreysoldtCorrection
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
        defect_locpot:
            The Locpot object for the supercell.
        bulk_locpot:
            The Locpot of the bulk supercell, note that since the locpot object is mutable,
            different defect entries can share the same bulk_locpot object.
            (Take care to not modify this object.)
        corrections:
            A dictionary of corrections to the energy.
    """

    defect: Defect
    charge_state: int
    sc_entry: ComputedStructureEntry
    sc_defect_fpos: Optional[ArrayLike] = None
    corrections: Optional[Dict[str, float]] = None

    def add_locpots(self, bulk_locpot: Locpot, defect_locpot: Locpot):
        """Add the bulk and defect locpot to the defect entry.

        Since we only need the correction values to reconstruct the defect entry
        and the Locpot objects are only used to compute the corrections once,
        it should not be serialized.
        Additionally, since the locpot object are mutable, different defect entries
        can share the same bulk_locpot object which also makes it a bad idea to serialize.
        """
        self.bulk_locpot = bulk_locpot
        self.defect_locpot = defect_locpot

        # get the defect position that should be used for freysoldt correction
        # if it is not already provided
        if self.sc_defect_fpos is None:
            finder = DefectSiteFinder()
            self.sc_defect_fpos = finder.get_defect_fpos(
                defect_structure=self.defect_locpot.structure,
                base_structure=self.bulk_locpot.structure,
            )

    def __post_init__(self):
        """Post-initialization."""
        self.charge_state = int(self.charge_state)
        self.corrections: dict = {} if self.corrections is None else self.corrections

    def _has_locpots(self):
        """Check if the bulk and defect locpots are available."""
        return all([hasattr(self, attr) for attr in ["bulk_locpot", "defect_locpot"]])

    def get_freysoldt_correction(self, dielectric_const: float | NDArray):
        """Calculate the Freysoldt correction.

        Returns:
            The Freysoldt correction.
        """
        if not self._has_locpots():
            raise ValueError("Locpots are not available. Please add them using the `add_locpots` method.")
        fc = FreysoldtCorrection(dielectric_const=dielectric_const)
        fc_correction = fc.get_correction(defect_entry=self, defect_frac_coords=self.sc_defect_fpos)
        self.corrections.update(fc_correction)  # type: ignore

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
    phase_digram: PhaseDiagram
    bulk_locpot: Locpot | None = None

    def __post_init__(self):
        """Post-initialization."""
        # reconstruct the phase diagram with the bulk entry
        entries = self.phase_digram.stable_entries | {self.bulk_entry}
        self.phase_digram = PhaseDiagram(entries)

    def vbm_formation_energy(self, defect_entry: DefectEntry, dep_elt: Element) -> tuple[float, float]:
        """Compute the formation energy at the VBM.

        Args:
            defect_entry:
                the defect entry for which the formation energy is computed.
            dep_elt:
                the dependent element for which the chemical potential is computed
                from the energy of the stable phase at the target composition.

        Returns:
            float:
                Formation energy for the situation where the dependent element is rare.
            float:
                Formation energy for the situation where the dependent element is abundant.
        """
        formation_en = (
            defect_entry.corrected_energy - self.bulk_entry.energy + float(defect_entry.charge_state) * self.vbm
        )

        defect: Defect = self.defect_entries[0].defect
        el_change = defect.element_changes
        el_poor_chempot, el_rich_chempot = self.critical_chemical_potential(dep_elt)

        elt_poor_res = formation_en
        elt_rich_res = formation_en
        for key, factor in el_change.items():
            elt_poor_res += el_poor_chempot[key] * factor
            elt_rich_res += el_rich_chempot[key] * factor

        return elt_poor_res, elt_rich_res

    def critical_chemical_potential(self, dep_elt: Element) -> Tuple[Dict[Element, float], Dict[Element, float]]:
        """Compute the critical chemical potentials.

        Args:
            dep_elt:
                the dependent element for which the chemical potential is computed
                from the energy of the stable phase at the target composition

        Returns:
            Dict[Element, float]:
                chemical potentials for the chase where the dep_elt is is rare.
                (e.g. O-poor growth conditions)
            Dict[Element, float]:
                chemical potentials for the chase where the dep_elt is is abundant.
                (e.g. O-rich growth conditions)
        """
        if self.phase_digram is None:
            raise RuntimeError("Phase diagram is not available.")
        pd = ensure_stable_bulk(self.phase_digram, self.bulk_entry)
        chem_pots = pd.getmu_vertices_stability_phase(self.bulk_entry.composition, dep_elt)
        dep_elt_poor = max(chem_pots, key=lambda x: x[dep_elt])
        dep_elt_rich = min(chem_pots, key=lambda x: x[dep_elt])
        return dep_elt_poor, dep_elt_rich


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


def get_transitions(lines: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Get the "transition" points in a list of lines.

    Given a list of lines represented as (m, b) pairs sorted in order of decreasing m.
    A "transition" point is a point where adjacent lines in the list intersect.
    i.e. intersection points (x_i, y_i) where line i intersects line i+1

    Args:
        lines: (m, b) format for each line

    Returns:
        List[List[float]]:
            List of intersection points.
    """
    # make sure the lines are sorted by decreasing slope
    lines = sorted(lines, key=lambda x: x[0], reverse=True)
    x_transitions = []
    for i, (m1, b1) in enumerate(lines[:-1]):
        m2, b2 = lines[i + 1]
        if m1 == m2:
            raise ValueError("The slopes (charge states) of the set of lines should be distinct.")
        x_transitions.append(((b2 - b1) / (m1 - m2), (m1 * b2 - m2 * b1) / (m1 - m2)))
    return x_transitions


def get_lower_envelope(lines):
    """Get the lower envelope of the formation energy.

    The lines are returned with decreasing slope.

    Args:
        lines: (m, b) format for each line

    Returns:
        List[List[float]]:
            List of intersection points.
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

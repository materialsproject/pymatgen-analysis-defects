"""Classes and methods related to thermodynamics and energy."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cdd
from monty.json import MSONable
from numpy.typing import ArrayLike, NDArray
from pymatgen.analysis.phase_diagram import PhaseDiagram
from pymatgen.core import Composition, Element
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
    dielectric: float | NDArray
    sc_defect_frac_coords: Optional[ArrayLike] = None
    corrections: Optional[Dict[str, float]] = None

    # def add_locpots(self, bulk_locpot: Locpot, defect_locpot: Locpot):
    #     """Add the bulk and defect locpot to the defect entry.

    #     Since we only need the correction values to reconstruct the defect entry
    #     and the Locpot objects are only used to compute the corrections once,
    #     it should not be serialized.
    #     Additionally, since the locpot object are mutable, different defect entries
    #     can share the same bulk_locpot object which also makes it a bad idea to serialize.
    #     """
    #     self.bulk_locpot = bulk_locpot
    #     self.defect_locpot = defect_locpot

    # get the defect position that should be used for freysoldt correction
    # if it is not already provided

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
    phase_diagram: PhaseDiagram

    def __post_init__(self):
        """Post-initialization."""
        # reconstruct the phase diagram with the bulk entry
        entries = self.phase_diagram.stable_entries | {self.bulk_entry}
        self.phase_diagram = PhaseDiagram(entries)

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

    def critical_chemical_potential_pmg(self, dep_elt: Element) -> Tuple[Dict[Element, float], Dict[Element, float]]:
        """Compute the critical chemical potentials.

        Using existing code in pymatgen.analysis.phase_diagram.

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
        if self.phase_diagram is None:
            raise RuntimeError("Phase diagram is not available.")
        pd = ensure_stable_bulk(self.phase_diagram, self.bulk_entry)
        chem_pots = pd.getmu_vertices_stability_phase(self.bulk_entry.composition, dep_elt)
        dep_elt_poor = max(chem_pots, key=lambda x: x[dep_elt])
        dep_elt_rich = min(chem_pots, key=lambda x: x[dep_elt])
        return dep_elt_poor, dep_elt_rich

    def critical_chemical_potential_cdd(self, dep_elt: Element) -> Tuple[Dict[Element, float], Dict[Element, float]]:
        """Compute the critical chemical potentials.

        Using the PyCDD library

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

    def lower_envelope(self, dep_elt: Element):
        """Compute the lower envelope of the formation energy diagram."""
        lines_elt_poor = []
        lines_elt_rich = []
        for defect_entry in self.defect_entries:
            elt_poor, elt_rich = self.vbm_formation_energy(defect_entry, dep_elt)
            lines_elt_poor.append((float(defect_entry.charge_state), elt_poor))
            lines_elt_rich.append((float(defect_entry.charge_state), elt_rich))

        trans_elt_poor = get_transitions(lines=lines_elt_poor)
        trans_elt_rich = get_transitions(lines=lines_elt_rich)

        return {
            "elt_poor": {
                "fermi_level": [l[0] for l in trans_elt_poor],
                "formation_energy": [l[1] for l in trans_elt_poor],
            },
            "elt_rich": {
                "fermi_level": [l[0] for l in trans_elt_rich],
                "formation_energy": [l[1] for l in trans_elt_rich],
            },
        }


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


def get_stability_region_cdd(phase_diagram: PhaseDiagram, composition: Composition):
    """Get the compositional stability boundary.

    Calculate the vertices of the stability region of a given composition.
    We define the stability region as the intersection of all half-spaces where:
        - the desired composition is stable
        - all other compositions are unstable
    The vertices are calculated using the double-description method implemented in pyCDD.

    Args:
        phase_diagram:
            Phase diagram, only need the hull data so it can be constructed from stable entries.
        composition:
            The target composition that must be stable

    Returns:
        list:
            The vertices of the stability region.
    """
    el_keys = sorted(phase_diagram.elements)
    stable_comps = {ient.composition.reduced_formula for ient in phase_diagram.stable_entries}

    if composition.reduced_formula not in stable_comps:
        raise RuntimeError("Composition needs to be a stable entry of the phase diagram.")

    ineqs = []
    for ient in phase_diagram.stable_entries:
        a_ = [-ient.composition[k] for k in el_keys]
        b_ = phase_diagram.get_form_energy(ient)
        # need A x <= b
        # input phase form: >= H(Phase)
        # all other phases do not form: <= H(Phase)
        if ient.composition.reduced_composition == composition.reduced_composition:
            b_ = -b_
            a_ = [-n_ for n_ in a_]
        ineqs.append([b_] + a_)

    ineq_mat = cdd.Matrix(ineqs, number_type="float")
    ineq_mat.rep_type = cdd.RepType.INEQUALITY

    poly = cdd.Polyhedron(ineq_mat)

    ext = poly.get_generators()
    res = []
    for a, *b in ext:
        if a == 1:
            res.append(dict(zip(map(str, el_keys), b)))
    return res


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

    Based on the fact that the lower envelope of the lines is
    given by the upper convex hull of the points (m, -b) as shown in:
    https://www.cs.umd.edu/class/spring2020/cmsc754/Lects/lect06-duality.pdf
    Note: The lines are returned with decreasing slope.

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

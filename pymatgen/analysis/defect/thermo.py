from __future__ import annotations

"""Classes and methods related to thermodynamics and energy."""

import logging
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

from monty.json import MSONable
from pymatgen.analysis.phase_diagram import PhaseDiagram
from pymatgen.core import Element, Structure
from pymatgen.entries.computed_entries import ComputedEntry, ComputedStructureEntry
from pymatgen.io.vasp.outputs import Locpot

from pymatgen.analysis.defect.core import Defect


@dataclass
class DefectEntry(MSONable):
    """Data for completed defect supercell calculation.

    Attributes:
        defect_type:
            Type of the defect (e.g. vacancy, substitution, etc.)
        defect:
            The defect object used to generate the supercell.
        structure:
            The supercell structure used to simulate the defect.
        charge_state:
            The charge state of the defect.
        energy:
            The total energy of the defect supercell.
        entry_id:
            The unique identifier for the defect entry.
    """

    # TODO: Ignore corrections until EVERYTHING else is working.
    defect: Defect
    charge_state: int
    energy: float
    entry_id: object | None = None
    num_sites: int = None
    structure: Structure | None = None

    def __post_init__(self):
        if self.structure is not None:
            if self.num_sites is not None and self.num_sites != self.structure.num_sites:
                raise ValueError(
                    f"The number of atoms in the structure ({self.structure.num_sites}) does not match the number of atoms ({self.num_sites})"
                )
            self.num_sites = len(self.structure)
        elif self.num_sites is None:
            raise ValueError("Must specify either structure or num_sites")


@dataclass
class FormationEnergyDiagram(MSONable):
    """Formation energy."""

    bulk_entry: ComputedStructureEntry
    defect_entries: List[DefectEntry]
    vbm: float
    phase_digram: PhaseDiagram | None = None
    bulk_locpot: Locpot | None = None
    defect_locpots: Iterable[Locpot] | None = None

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
        formation_en = defect_entry.energy - self.bulk_entry.energy + self.vbm
        formation_en += self.correction
        defect: Defect = self.defect_entries[0].defect
        el_change = defect.element_changes
        el_poor_chempot, el_rich_chempot = self.critical_chemical_potential(dep_elt)

        elt_poor_res = formation_en
        elt_rich_res = formation_en
        for key, factor in el_change.items():
            elt_poor_res += el_poor_chempot[key] * factor
            elt_rich_res += el_rich_chempot[key] * factor
        return elt_poor_res, elt_rich_res

    @property
    def correction(self) -> float:
        """The correction to the formation energy: Example finite-size corrections."""
        return 0

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

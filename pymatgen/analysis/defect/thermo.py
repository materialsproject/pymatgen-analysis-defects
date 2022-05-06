from __future__ import annotations

"""Classes and methods related to thermodynamics and energy."""

import logging
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

from monty.json import MSONable
from pymatgen.analysis.phase_diagram import PhaseDiagram
from pymatgen.core import Element
from pymatgen.entries.computed_entries import ComputedEntry, ComputedStructureEntry
from pymatgen.io.vasp.outputs import Locpot

from pymatgen.analysis.defect.core import Defect


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
        corrections:
            A dictionary of corrections to the energy.
    """

    defect: Defect
    charge_state: int
    sc_entry: ComputedStructureEntry
    corrections: Dict[str, float] = None


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
        formation_en = (
            defect_entry.sc_entry.energy
            - self.bulk_entry.energy
            + defect_entry.charge_state * self.vbm
            + self.correction
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

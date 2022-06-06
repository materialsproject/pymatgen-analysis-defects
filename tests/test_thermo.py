import pytest

from pymatgen.analysis.defect.corrections import plot_plnr_avg
from pymatgen.analysis.defect.thermo import (
    DefectEntry,
    get_lower_envelope,
    get_transitions,
)


def test_lower_envelope():
    # Test the lower envelope and transition code with a simple example
    lines = [[4, 12], [-1, 3], [-5, 4], [-2, 1], [3, 8], [-4, 14], [2, 12], [3, 8]]
    lower_envelope_ref = [(4, 12), (3, 8), (-2, 1), (-5, 4)]  # answer from visual inspection (ordered)
    transitions_ref = [(-4, -4), (-1.4, 3.8), (1, -1)]
    lower_envelope = get_lower_envelope(lines)
    assert lower_envelope == lower_envelope_ref
    assert get_transitions(lower_envelope) == transitions_ref


def test_defect_entry(data_Mg_Ga, defect_Mg_Ga):
    bulk_locpot = data_Mg_Ga["bulk_sc"]["locpot"]

    def get_data(q):
        computed_entry = data_Mg_Ga[f"q={q}"]["vasprun"].get_computed_entry(inc_structure=True)
        defect_locpot = data_Mg_Ga[f"q={q}"]["locpot"]

        def_entry = DefectEntry(defect=defect_Mg_Ga, charge_state=q, sc_entry=computed_entry, dielectric=14)
        plot_data = def_entry.get_freysoldt_correction(defect_locpot=defect_locpot, bulk_locpot=bulk_locpot)
        return def_entry, plot_data

    def_entry, plot_data = get_data(0)
    assert def_entry.corrections["freysoldt_electrostatic"] == pytest.approx(0.00, abs=1e-4)
    assert def_entry.corrections["freysoldt_potential_alignment"] == pytest.approx(0.00, abs=1e-4)

    def_entry, plot_data = get_data(-2)
    assert def_entry.corrections["freysoldt_electrostatic"] > 0
    assert def_entry.corrections["freysoldt_potential_alignment"] > 0

    def_entry, plot_data = get_data(1)
    assert def_entry.corrections["freysoldt_electrostatic"] > 0
    assert def_entry.corrections["freysoldt_potential_alignment"] < 0

    plot_plnr_avg(plot_data[0])


def test_free_energy(data_Mg_Ga, defect_Mg_Ga):
    bulk_locpot = data_Mg_Ga["bulk_sc"]["locpot"]

    def get_data(q):
        computed_entry = data_Mg_Ga[f"q={q}"]["vasprun"].get_computed_entry(inc_structure=True)
        defect_locpot = data_Mg_Ga[f"q={q}"]["locpot"]

        def_entry = DefectEntry(defect=defect_Mg_Ga, charge_state=q, sc_entry=computed_entry, dielectric=14)
        plot_data = def_entry.get_freysoldt_correction(defect_locpot=defect_locpot, bulk_locpot=bulk_locpot)
        return def_entry, plot_data

    bulk_entry = data_Mg_Ga["bulk_sc"]["vasprun"].get_computed_entry(inc_structure=True)
    bulk_locpot = data_Mg_Ga["bulk_sc"]["locpot"]

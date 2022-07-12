import numpy as np
import pytest

from pymatgen.analysis.defects.corrections import plot_plnr_avg
from pymatgen.analysis.defects.thermo import (
    FormationEnergyDiagram,
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
    assert get_transitions(lower_envelope, -5, 2) == [(-5, -8)] + transitions_ref + [(2, -6)]


def test_defect_entry2(defect_entries_Mg_Ga):
    defect_entries, plot_data = defect_entries_Mg_Ga

    def_entry = defect_entries[0]
    assert def_entry.corrections["freysoldt_electrostatic"] == pytest.approx(0.00, abs=1e-4)
    assert def_entry.corrections["freysoldt_potential_alignment"] == pytest.approx(0.00, abs=1e-4)

    def_entry = defect_entries[-2]
    assert def_entry.corrections["freysoldt_electrostatic"] > 0
    assert def_entry.corrections["freysoldt_potential_alignment"] > 0

    def_entry = defect_entries[1]
    assert def_entry.corrections["freysoldt_electrostatic"] > 0
    assert def_entry.corrections["freysoldt_potential_alignment"] < 0

    # test that the plotting code runs
    plot_plnr_avg(plot_data[0][1])


def test_formation_energy(data_Mg_Ga, defect_entries_Mg_Ga, stable_entries_Mg_Ga_N):
    bulk_vasprun = data_Mg_Ga["bulk_sc"]["vasprun"]
    bulk_bs = bulk_vasprun.get_band_structure()
    vbm = bulk_bs.get_vbm()["energy"]
    bulk_entry = bulk_vasprun.get_computed_entry(inc_structure=False)
    defect_entries, plot_data = defect_entries_Mg_Ga

    def_ent_list = list(defect_entries.values())

    fed = FormationEnergyDiagram(
        bulk_entry=bulk_entry,
        defect_entries=def_ent_list,
        vbm=vbm,
        pd_entries=stable_entries_Mg_Ga_N,
        inc_inf_value=True,
    )
    assert len(fed.chempot_limits) == 4

    fed = FormationEnergyDiagram(
        bulk_entry=bulk_entry,
        defect_entries=def_ent_list,
        vbm=vbm,
        pd_entries=stable_entries_Mg_Ga_N,
        inc_inf_value=False,
    )
    assert len(fed.chempot_limits) == 2

    # check that the shape of the formation energy diagram does not change
    cp_dict = fed.chempot_limits[0]
    form_en = np.array(fed.get_transitions(cp_dict, 0, 5))
    x_ref = form_en[:, 0]
    y_ref = form_en[:, 1]
    y_ref = y_ref - y_ref.min()

    for point in fed.chempot_limits:
        form_en = np.array(fed.get_transitions(point, 0, 5))
        x = form_en[:, 0]
        y = form_en[:, 1]
        y = y - y.min()
        assert np.allclose(x, x_ref)
        assert np.allclose(y, y_ref)

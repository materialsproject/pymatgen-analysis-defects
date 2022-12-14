import numpy as np
import pytest
from pymatgen.analysis.phase_diagram import PhaseDiagram

from pymatgen.core import PeriodicSite
from pymatgen.analysis.defects.core import Interstitial
from pymatgen.analysis.defects.corrections.freysoldt import plot_plnr_avg
from pymatgen.analysis.defects.thermo import (
    DefectEntry,
    FormationEnergyDiagram,
    MultiFormationEnergyDiagram,
    get_lower_envelope,
    get_transitions,
)


def test_lower_envelope():
    # Test the lower envelope and transition code with a simple example
    lines = [[4, 12], [-1, 3], [-5, 4], [-2, 1], [3, 8], [-4, 14], [2, 12], [3, 8]]
    lower_envelope_ref = [
        (4, 12),
        (3, 8),
        (-2, 1),
        (-5, 4),
    ]  # answer from visual inspection (ordered)
    transitions_ref = [(-4, -4), (-1.4, 3.8), (1, -1)]
    lower_envelope = get_lower_envelope(lines)
    assert lower_envelope == lower_envelope_ref
    assert get_transitions(lower_envelope, -5, 2) == [(-5, -8)] + transitions_ref + [
        (2, -6)
    ]


def test_defect_entry(defect_entries_Mg_Ga):
    defect_entries, plot_data = defect_entries_Mg_Ga

    def_entry = defect_entries[0]
    assert def_entry.corrections["electrostatic"] == pytest.approx(0.00, abs=1e-4)
    assert def_entry.corrections["potential_alignment"] == pytest.approx(0.00, abs=1e-4)

    def_entry = defect_entries[-2]
    assert def_entry.corrections["electrostatic"] > 0
    assert def_entry.corrections["potential_alignment"] > 0

    def_entry = defect_entries[1]
    assert def_entry.corrections["electrostatic"] > 0
    assert def_entry.corrections["potential_alignment"] > 0

    # test that the plotting code runs
    plot_plnr_avg(plot_data[0][1])
    plot_plnr_avg(defect_entries[1].correction_metadata[1])

    vr1 = plot_data[0][1]["pot_plot_data"]["Vr"]
    vr2 = defect_entries[0].correction_metadata[1]["pot_plot_data"]["Vr"]
    assert np.allclose(vr1, vr2)


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
        inc_inf_values=True,
    )
    assert len(fed.chempot_limits) == 4

    fed = FormationEnergyDiagram(
        bulk_entry=bulk_entry,
        defect_entries=def_ent_list,
        vbm=vbm,
        pd_entries=stable_entries_Mg_Ga_N,
        inc_inf_values=False,
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

    # test the constructor with materials project phase diagram
    atomic_entries = list(
        filter(lambda x: len(x.composition.elements) == 1, stable_entries_Mg_Ga_N)
    )
    pd = PhaseDiagram(stable_entries_Mg_Ga_N)
    fed = FormationEnergyDiagram.with_atomic_entries(
        bulk_entry=bulk_entry,
        defect_entries=def_ent_list,
        atomic_entries=atomic_entries,
        vbm=vbm,
        inc_inf_values=False,
        phase_diagram=pd,
    )

    assert len(fed.chempot_limits) == 2


def test_multi(data_Mg_Ga, defect_entries_Mg_Ga, stable_entries_Mg_Ga_N):
    bulk_vasprun = data_Mg_Ga["bulk_sc"]["vasprun"]
    bulk_dos = bulk_vasprun.complete_dos
    _, vbm = bulk_dos.get_cbm_vbm()
    bulk_entry = bulk_vasprun.get_computed_entry(inc_structure=False)
    defect_entries, plot_data = defect_entries_Mg_Ga
    def_ent_list = list(defect_entries.values())

    with pytest.raises(ValueError, match="Defects are not of same type! Use MultiFormationEnergyDiagram for multiple defect types"):
        inter = Interstitial(
        structure=defect_entries[0].defect.structure,
        site=PeriodicSite("H", [0,0,0], defect_entries[0].defect.structure.lattice)
        )
        fake_defect_entry = DefectEntry(defect=inter, sc_entry=defect_entries[0].sc_entry, charge_state=0)
        FormationEnergyDiagram(
            bulk_entry=bulk_entry,
            defect_entries=def_ent_list + [fake_defect_entry],
            vbm=vbm,
            pd_entries=stable_entries_Mg_Ga_N,
            inc_inf_values=False,
            )

    fed = FormationEnergyDiagram(
        bulk_entry=bulk_entry,
        defect_entries=def_ent_list,
        vbm=vbm,
        pd_entries=stable_entries_Mg_Ga_N,
        inc_inf_values=False,
    )
    mfed = MultiFormationEnergyDiagram(formation_energy_diagrams=[fed])
    ef = mfed.solve_for_fermi_level(chempots=mfed.chempot_limits[0], temperature=300, dos=bulk_dos)
    assert ef == pytest.approx(0.6986374710290937, 1e-3)


def test_formation_from_directory(test_dir, stable_entries_Mg_Ga_N, defect_Mg_Ga):
    sc_dir = test_dir / "Mg_Ga"
    qq = []
    for q in [-1, 0, 1]:
        qq.append(q)
        dmap = {"bulk": sc_dir / "bulk_sc"}
        dmap.update(zip(qq, map(lambda x: sc_dir / f"q={x}", qq)))
        assert len(dmap) == len(qq) + 1
        fed = FormationEnergyDiagram.with_directories(
            directory_map=dmap,
            defect=defect_Mg_Ga,
            pd_entries=stable_entries_Mg_Ga_N,
            dielectric=10,
        )
        trans = fed.get_transitions(fed.chempot_limits[1], x_min=-100, x_max=100)
        assert len(trans) == 1 + len(qq)

import numpy as np
import pytest
from pymatgen.analysis.defects.core import Interstitial, PeriodicSite, Vacancy
from pymatgen.analysis.defects.utils import (
    ChargeInsertionAnalyzer,
    TopographyAnalyzer,
    cluster_nodes,
    get_avg_chg,
    get_local_extrema,
    get_localized_states,
    get_plane_spacing,
    group_docs,
)
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core.periodic_table import Specie
from pymatgen.io.vasp.outputs import Chgcar


def test_get_local_extrema(gan_struct):
    data = np.ones((48, 48, 48))
    chgcar = Chgcar(poscar=gan_struct, data={"total": data})
    frac_pos = [[0, 0, 0], [0.25, 0.25, 0.25], [0.5, 0.5, 0.5], [0.75, 0.75, 0.75]]
    for fpos in frac_pos:
        idx = np.multiply(fpos, chgcar.data["total"].shape).astype(int)
        chgcar.data["total"][idx[0], idx[1], idx[2]] = 0
    loc_min = get_local_extrema(chgcar, frac_pos)
    for a, b in zip(sorted(loc_min.tolist()), sorted(frac_pos)):
        assert np.allclose(a, b)


def test_cluster_nodes(gan_struct):
    frac_pos = [
        [0, 0, 0],
        [0.25, 0.25, 0.25],
        [0.5, 0.5, 0.5],
        [0.75, 0.75, 0.75],
    ]
    added = [
        [0.0002, 0.0001, 0.0001],
        [0.0002, 0.0002, 0.0003],
        [0.25001, 0.24999, 0.24999],
        [0.25, 0.249999, 0.250001],
    ]  # all the displacements are positive so we dont have to worry about periodic boundary conditions
    clusters = cluster_nodes(frac_pos + added, gan_struct.lattice)

    for a, b in zip(sorted(clusters.tolist()), sorted(frac_pos)):
        assert np.allclose(a, b, atol=0.001)


def test_get_avg_chg(gan_struct):
    data = np.ones((48, 48, 48))
    chgcar = Chgcar(poscar=gan_struct, data={"total": data})
    fpos = [0.1, 0.1, 0.1]
    avg_chg_sphere = get_avg_chg(chgcar, fpos)
    avg_chg = np.sum(chgcar.data["total"]) / chgcar.ngridpts / chgcar.structure.volume
    pytest.approx(avg_chg_sphere, avg_chg)


def test_chgcar_insertion(chgcar_fe3o4):
    chgcar = chgcar_fe3o4
    insert_ref = [
        (
            0.03692438178614583,
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.5], [0.0, 0.5, 0.0], [0.5, 0.0, 0.0]],
        ),  # corners and edge centers
        (
            0.10068764899215804,
            [[0.375, 0.375, 0.375], [0.625, 0.625, 0.625]],
        ),  # center of Fe-O cages
    ]
    cia = ChargeInsertionAnalyzer(chgcar)
    insert_groups = cia.filter_and_group(max_avg_charge=0.5)
    for (avg_chg, group), (ref_chg, ref_fpos) in zip(insert_groups, insert_ref):
        fpos = sorted(group)
        pytest.approx(avg_chg, ref_chg)
        assert np.allclose(fpos, ref_fpos)


def test_topography_analyzer(chgcar_fe3o4):
    struct = chgcar_fe3o4.structure
    ta = TopographyAnalyzer(struct, ["Fe", "O"], [], check_volume=True)
    node_struct = ta.get_structure_with_nodes()
    # All sites with species X
    dummy_sites = [site for site in node_struct if site.specie.symbol == "X"]
    assert len(dummy_sites) == 100

    # Check value error
    with pytest.raises(ValueError):
        ta = TopographyAnalyzer(struct, ["O"], ["Fe"], check_volume=True)


def test_get_localized_states(v_ga):
    vaspruns = v_ga[(0, -1)]["vaspruns"]
    procar = v_ga[(0, -1)]["procar"]
    vr = vaspruns[1]
    bs = vr.get_band_structure()
    res = get_localized_states(bs, procar=procar)
    loc_bands = set()
    for iband, ikpt, ispin, val in get_localized_states(bs, procar=procar):
        loc_bands.add(iband)
    assert loc_bands == {
        138,
    }

    vaspruns = v_ga[(-1, 0)]["vaspruns"]
    procar = v_ga[(-1, 0)]["procar"]
    vr = vaspruns[1]
    bs = vr.get_band_structure()

    loc_bands = set()
    for iband, ikpt, ispin, val in get_localized_states(
        bs, procar=procar, band_window=100
    ):
        loc_bands.add(iband)
    assert loc_bands == {75, 77}  # 75 and 77 are more localized core states


def test_group_docs(gan_struct):
    s = gan_struct.copy()
    vac1 = Vacancy(s, s.sites[0])
    vac2 = Vacancy(s, s.sites[1])
    vac3 = Vacancy(s, s.sites[2])
    vac4 = Vacancy(s, s.sites[3])

    def get_interstitial(fpos):
        n_site = PeriodicSite(Specie("N"), fpos, s.lattice)
        return Interstitial(s, n_site)

    # two interstitials are at inequivalent sites so should be in different groups
    int1 = get_interstitial([0.0, 0.0, 0.0])
    int2 = get_interstitial([0.0, 0.0, 0.25])
    sm = StructureMatcher()
    # Test that the grouping works without a key function (only structure)
    sgroups = group_docs(
        [vac1, vac2, int1, vac3, vac4, int2],
        sm,
        lambda x: x.defect_structure,
    )
    res = []
    for _, group in sgroups:
        defect_names = ",".join([x.name for x in group])
        res.append(defect_names)
    # the final sorted groups
    assert "|".join(sorted(res)) == "N_i|N_i|v_Ga,v_Ga|v_N,v_N"

    # Test that the grouping works with a key function (structure and name)
    sgroups = group_docs(
        [vac1, vac2, int1, vac3, vac4, int1, int2],
        sm,
        lambda x: x.defect_structure,
        lambda x: x.name,
    )
    res = []
    g_names = []
    for name, group in sgroups:
        defect_names = ",".join([x.name for x in group])
        g_names.append(name)
        res.append(defect_names)
    assert "|".join(sorted(res)) == "N_i|N_i,N_i|v_Ga,v_Ga|v_N,v_N"
    assert "|".join(sorted(g_names)) == "N_i:0|N_i:1|v_Ga|v_N"


def test_plane_spacing(gan_struct):
    lattice = gan_struct.lattice.matrix
    assert np.allclose(get_plane_spacing(lattice), [2.785, 2.785, 5.239], atol=0.001)

import numpy as np
import pytest
from pymatgen.io.vasp.outputs import Chgcar

from pymatgen.analysis.defects.utils import (
    ChargeInsertionAnalyzer,
    cluster_nodes,
    get_avg_chg,
    get_local_extrema,
)


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

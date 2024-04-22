import numpy as np
from monty.serialization import loadfn
from pymatgen.analysis.defects.generators import VacancyGenerator
from pymatgen.analysis.defects.supercells import (
    _ase_cubic,
    get_matched_structure_mapping,
    get_sc_fromstruct,
    get_closest_sc_mat
)
import pytest


def test_supercells(gan_struct):
    uc = gan_struct.copy()
    sc_mat = get_sc_fromstruct(uc)
    sc = uc * sc_mat
    assert sc_mat.shape == (3, 3)

    sc_mat2, _ = get_matched_structure_mapping(uc, sc)
    assert sc_mat2.shape == (3, 3)
    sc2 = uc * sc_mat2
    np.testing.assert_allclose(
        sc.lattice.abc, sc2.lattice.abc
    )  # the sc_mat can be reconstructed from the sc


def test_ase_supercells(gan_struct):
    sc_mat = _ase_cubic(gan_struct, min_atoms=4, max_atoms=8, min_length=1.0)
    sc = gan_struct * sc_mat
    assert 4 <= sc.num_sites <= 8

    # check raise
    with pytest.raises(RuntimeError):
        _ase_cubic(gan_struct, min_atoms=4, max_atoms=8, min_length=10)


def test_closest_sc_mat(test_dir):
    si_o_structs = loadfn(test_dir / "Si-O_structs.json")
    ref_sc_mat = [[2,1,2], [2,0,3], [2,1,1]]
    
    vg = VacancyGenerator()
    def get_vac(s, sc_mat):
        vac = next(vg.generate(s, rm_species=["O"]))
        return vac.get_supercell_structure(sc_mat=sc_mat)
    
    def check_uc(uc_struct, sc_mat):
        vac_sc = get_vac(uc_struct, sc_mat)
        sorted_results = get_closest_sc_mat(uc_struct, vac_sc, debug=True)
        min_dist = sorted_results[0][0]
        close_mats = [r[2] for r in sorted_results if r[0] < min_dist*1.1]
        is_matched = [np.allclose(ref_sc_mat, x) for x in close_mats]
        assert any(is_matched)
    
    for s in si_o_structs:
        check_uc(s, ref_sc_mat)

    uc_struct = si_o_structs[0]
    vac_struct = get_vac(uc_struct, ref_sc_mat)
    res = get_closest_sc_mat(uc_struct=uc_struct, sc_struct=vac_struct, debug=False)
    assert np.allclose(res, ref_sc_mat)
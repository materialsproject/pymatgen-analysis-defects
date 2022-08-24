import numpy as np

from pymatgen.analysis.defects.supercells import (
    _ase_cubic,
    get_matched_structure_mapping,
    get_sc_fromstruct,
)


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
    sc_mat = _ase_cubic(gan_struct, min_atoms=4, max_atoms=8)
    sc = gan_struct * sc_mat
    assert 4 <= sc.num_sites <= 8

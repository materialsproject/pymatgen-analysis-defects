import numpy as np

from pymatgen.analysis.defects2.supercells import (
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
    np.testing.assert_allclose(sc.lattice.abc, sc2.lattice.abc)  # the sc_mat can be reconstructed from the sc

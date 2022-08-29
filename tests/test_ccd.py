from collections import namedtuple

import numpy as np
import pytest

from pymatgen.analysis.defects.ccd import _get_wswq_slope


def test_HarmonicDefect(v_ga):
    from pymatgen.analysis.defects.ccd import HarmonicDefect

    vaspruns = v_ga[(0, -1)]["vaspruns"]
    procar = v_ga[(0, -1)]["procar"]
    hd0 = HarmonicDefect.from_vaspruns(vaspruns, charge_state=0)
    pytest.approx(hd0.distortions[1], 0.0)
    pytest.approx(hd0.omega_eV, 0.032680)
    wswqs = v_ga[(0, -1)]["wswqs"]
    relaxed_bs = vaspruns[1].get_band_structure()
    elph_me = hd0.get_elph_me(bandstructure=relaxed_bs, wswqs=wswqs, procar=procar)
    assert np.allclose(elph_me[..., 138], 0.0)  # ediff should be zero for defect band
    assert np.linalg.norm(elph_me[..., 139]) > 0


def test_wswq_slope():
    mats = [np.ones((3, 3)), np.zeros((3, 3)), np.ones((3, 3))]
    FakeWSWQ = namedtuple("FakeWSWQ", ["data"])
    fake_wswqs = [FakeWSWQ(data=m) for m in mats]

    res = _get_wswq_slope([-0.5, 0, 0.5], fake_wswqs)
    np.allclose(res, np.ones((3, 3)) * 2)

    res = _get_wswq_slope([1.0, 0, -1.0], fake_wswqs)
    np.allclose(res, np.ones((3, 3)) * -1)

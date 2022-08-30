from collections import namedtuple

import numpy as np
import pytest

from pymatgen.analysis.defects.ccd import _get_wswq_slope


def test_HarmonicDefect(v_ga):
    from pymatgen.analysis.defects.ccd import HarmonicDefect

    vaspruns = v_ga[(0, -1)]["vaspruns"]
    procar = v_ga[(0, -1)]["procar"]
    hd0 = HarmonicDefect.from_vaspruns(vaspruns, charge_state=0, procar=procar)
    pytest.approx(hd0.distortions[1], 0.0)
    pytest.approx(hd0.omega_eV, 0.032680)
    wswqs = v_ga[(0, -1)]["wswqs"]
    # check for ValueError
    with pytest.raises(ValueError):
        elph_me = hd0.get_elph_me(wswqs=wswqs)

    hd0 = HarmonicDefect.from_vaspruns(
        vaspruns, charge_state=0, procar=procar, store_bandstructure=True
    )
    elph_me = hd0.get_elph_me(wswqs=wswqs)
    assert np.allclose(elph_me[..., 138], 0.0)  # ediff should be zero for defect band
    assert np.linalg.norm(elph_me[..., 139]) > 0


def test_OpticalHarmonicDefect(v_ga):
    from pymatgen.analysis.defects.ccd import OpticalHarmonicDefect

    vaspruns = v_ga[(0, -1)]["vaspruns"]
    procar = v_ga[(0, -1)]["procar"]
    wavder = v_ga[(0, -1)]["waveder"]
    hd0 = OpticalHarmonicDefect.from_vaspruns_and_waveder(
        vaspruns,
        waveder=wavder,
        charge_state=0,
        procar=procar,
    )

    # the non-optical part should behave the same
    wswqs = v_ga[(0, -1)]["wswqs"]
    elph_me = hd0.get_elph_me(wswqs=wswqs)
    assert np.allclose(elph_me[..., 138], 0.0)  # ediff should be zero for defect band
    assert np.linalg.norm(elph_me[..., 139]) > 0

    # # check that waveder is symmetric
    def is_symm(waveder, i, j):
        assert (
            np.max(
                np.abs(
                    np.abs(waveder.cder_data[i, j, :, :])
                    - np.abs(waveder.cder_data[j, i, :, :])
                )
            )
            <= 1e-10
        )

    is_symm(hd0.waveder, 123, 42)
    is_symm(hd0.waveder, 138, 69)

    nbands_spectra, *_ = hd0._get_spectra().shape
    nbands_dipole, *_ = hd0._get_defect_dipoles().shape
    assert nbands_spectra == nbands_dipole


def test_wswq_slope():
    mats = [np.ones((3, 5)), np.zeros((3, 5)), np.ones((3, 5))]
    FakeWSWQ = namedtuple("FakeWSWQ", ["data"])
    fake_wswqs = [FakeWSWQ(data=m) for m in mats]

    res = _get_wswq_slope([-0.5, 0, 0.5], fake_wswqs)
    np.allclose(res, np.ones((3, 5)) * 2)

    res = _get_wswq_slope([1.0, 0, -1.0], fake_wswqs)
    np.allclose(res, np.ones((3, 5)) * -1)

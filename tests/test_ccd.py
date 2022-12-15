from collections import namedtuple

import numpy as np
import pytest

from pymatgen.analysis.defects.ccd import _get_wswq_slope, get_dQ


def test_HarmonicDefect(v_ga):
    from pymatgen.analysis.defects.ccd import HarmonicDefect

    vaspruns = v_ga[(0, -1)]["vaspruns"]
    procar = v_ga[(0, -1)]["procar"]
    hd0 = HarmonicDefect.from_vaspruns(
        vaspruns,
        charge_state=0,
        procar=procar,
    )
    assert hd0.spin_index == 1
    pytest.approx(hd0.distortions[1], 0.0)
    pytest.approx(hd0.omega_eV, 0.032680)
    wswqs = v_ga[(0, -1)]["wswqs"]

    # check for ValueError
    with pytest.raises(ValueError):
        elph_me = hd0.get_elph_me(wswqs=wswqs)

    hd0 = HarmonicDefect.from_vaspruns(
        vaspruns,
        charge_state=0,
        procar=procar,
        store_bandstructure=True,
    )
    elph_me = hd0.get_elph_me(wswqs=wswqs)
    assert np.allclose(elph_me[..., 138], 0.0)  # ediff should be zero for defect band
    assert np.linalg.norm(elph_me[..., 139]) > 0

    hd2 = HarmonicDefect.from_vaspruns(
        vaspruns, charge_state=0, procar=procar, defect_band=((139, 0, 1), (139, 1, 1))
    )
    assert hd2.spin_index == 1

    # check for ValueError
    with pytest.raises(ValueError) as e:
        hd3 = HarmonicDefect.from_vaspruns(
            vaspruns,
            charge_state=0,
            procar=procar,
            defect_band=((139, 0, 1), (139, 1, 0)),
        )
        hd3.spin
    assert "Spin index" in str(e.value)


# def test_OpticalHarmonicDefect(v_ga):
#     from pymatgen.analysis.defects.ccd import OpticalHarmonicDefect

#     vaspruns = v_ga[(0, -1)]["vaspruns"]
#     procar = v_ga[(0, -1)]["procar"]
#     wavder = v_ga[(0, -1)]["waveder"]
#     hd0 = OpticalHarmonicDefect.from_vaspruns_and_waveder(
#         vaspruns,
#         waveder=wavder,
#         charge_state=0,
#         procar=procar,
#     )

#     # the non-optical part should behave the same
#     wswqs = v_ga[(0, -1)]["wswqs"]
#     elph_me = hd0.get_elph_me(wswqs=wswqs)
#     assert np.allclose(elph_me[..., 138], 0.0)  # ediff should be zero for defect band
#     assert np.linalg.norm(elph_me[..., 139]) > 0

#     # # check that waveder is symmetric
#     def is_symm(waveder, i, j):
#         assert (
#             np.max(
#                 np.abs(
#                     np.abs(waveder.cder_data[i, j, :, :])
#                     - np.abs(waveder.cder_data[j, i, :, :])
#                 )
#             )
#             <= 1e-10
#         )

#     is_symm(hd0.waveder, 123, 42)
#     is_symm(hd0.waveder, 138, 69)

#     nbands_spectra, *_ = hd0._get_spectra().shape
#     nbands_dipole, *_ = hd0._get_defect_dipoles().shape
#     assert nbands_spectra == nbands_dipole


def test_wswq_slope():
    mats = [np.ones((3, 5)), np.zeros((3, 5)), np.ones((3, 5))]
    FakeWSWQ = namedtuple("FakeWSWQ", ["data"])
    fake_wswqs = [FakeWSWQ(data=m) for m in mats]

    res = _get_wswq_slope([-0.5, 0, 0.5], fake_wswqs)
    np.allclose(res, np.ones((3, 5)) * 2)

    res = _get_wswq_slope([1.0, 0, -1.0], fake_wswqs)
    np.allclose(res, np.ones((3, 5)) * -1)


def test_SRHCapture(v_ga):
    from pymatgen.analysis.defects.ccd import HarmonicDefect, SRHCapture

    vaspruns = v_ga[(0, -1)]["vaspruns"]
    procar = v_ga[(0, -1)]["procar"]
    hd0 = HarmonicDefect.from_vaspruns(
        vaspruns, charge_state=0, procar=procar, store_bandstructure=True
    )

    vaspruns = v_ga[(-1, 0)]["vaspruns"]
    procar = v_ga[(-1, 0)]["procar"]
    hdm1 = HarmonicDefect.from_vaspruns(
        vaspruns, charge_state=-1, procar=procar, store_bandstructure=True
    )
    dQ = get_dQ(hd0.structures[hd0.relaxed_index], hdm1.structures[hdm1.relaxed_index])
    srh_cap = SRHCapture(hd0, hdm1, dQ=dQ, wswqs=v_ga[(0, -1)]["wswqs"])

    c_n = srh_cap.get_coeff(
        T=[100, 200, 300],
        dE=1.0,
        volume=hd0.structures[hd0.relaxed_index].volume,
        kpt_index=1,
    )
    ref_results = [1.89187260e-34, 6.21019152e-33, 3.51501688e-31]
    assert np.allclose(c_n, ref_results)


def test_SRHCapture_from_dir(test_dir):
    from pymatgen.analysis.defects.ccd import SRHCapture

    DEF_DIR = test_dir / "v_Ga"
    srh = SRHCapture.from_directories(
        initial_dirs=[DEF_DIR / "ccd_0_-1" / str(i) for i in [0, 1, 2]],
        final_dirs=[DEF_DIR / "ccd_-1_0" / str(i) for i in [0, 1, 2]],
        wswq_dir=DEF_DIR / "ccd_0_-1" / "wswqs",
        store_bandstructure=True,
    )
    c_n = srh.get_coeff(
        T=[100, 200, 300],
        dE=1.0,
        kpt_index=1,
    )
    ref_results = [1.89187260e-34, 6.21019152e-33, 3.51501688e-31]
    assert np.allclose(c_n, ref_results)

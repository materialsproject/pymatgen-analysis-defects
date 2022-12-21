from collections import namedtuple

import numpy as np
import pytest

from pymatgen.analysis.defects.ccd import HarmonicDefect, Waveder, _get_wswq_slope


@pytest.fixture(scope="session")
def hd0(v_ga):
    vaspruns = v_ga[(0, -1)]["vaspruns"]
    procar = v_ga[(0, -1)]["procar"]
    hd0 = HarmonicDefect.from_vaspruns(
        vaspruns,
        charge_state=0,
        procar=procar,
        store_bandstructure=True,
    )
    assert hd0.spin_index == 1
    assert pytest.approx(hd0.distortions[1]) == 0.0
    assert pytest.approx(hd0.omega_eV) == 0.03268045792725
    assert hd0.defect_band == [(138, 0, 1), (138, 1, 1)]
    return hd0


@pytest.fixture(scope="session")
def hd1(v_ga):
    vaspruns = v_ga[(-1, 0)]["vaspruns"]
    procar = v_ga[(-1, 0)]["procar"]
    hd1 = HarmonicDefect.from_vaspruns(
        vaspruns,
        charge_state=1,
        procar=procar,
        store_bandstructure=True,
    )
    assert pytest.approx(hd1.omega_eV) == 0.03341323356861477
    return hd1


def test_HarmonicDefect(hd0, v_ga, test_dir):
    # test other basic reading functions for HarmonicDefect
    vaspruns = v_ga[(0, -1)]["vaspruns"]
    procar = v_ga[(0, -1)]["procar"]
    hd0 = HarmonicDefect.from_vaspruns(
        vaspruns,
        charge_state=0,
        procar=procar,
        store_bandstructure=True,
    )
    assert hd0.defect_band == [(138, 0, 1), (138, 1, 1)]

    hd0p = HarmonicDefect.from_directories(
        directories=[test_dir / "v_Ga" / "ccd_0_-1" / str(i) for i in range(3)],
        charge_state=0,
    )
    assert hd0p.defect_band == [(138, 0, 1), (138, 1, 1)]

    hd2 = HarmonicDefect.from_vaspruns(
        vaspruns, charge_state=0, procar=procar, defect_band=((139, 0, 1), (139, 1, 1))
    )
    assert hd2.spin_index == 1

    vaspruns = v_ga[(0, -1)]["vaspruns"]
    procar = v_ga[(0, -1)]["procar"]
    # check for ValueError when you have non-unique spin for the defect band
    with pytest.raises(ValueError) as e:
        hd3 = HarmonicDefect.from_vaspruns(
            vaspruns,
            charge_state=0,
            procar=procar,
            defect_band=((139, 0, 1), (139, 1, 0)),
        )
        hd3.spin
    assert "Spin index" in str(e.value)


def test_wswq(hd0, test_dir):
    wswq_dir = test_dir / "v_Ga" / "ccd_0_-1" / "wswqs"

    # check for ValueError when you have mis-matched distortions and wswqs
    with pytest.raises(ValueError) as e:
        hd0.read_wswqs(
            directory=wswq_dir,
            distortions=[
                1,
            ],
        )
    assert "distortions" in str(e.value)

    hd0.read_wswqs(wswq_dir)
    elph_me = hd0.get_elph_me((138, 1, 1))
    assert np.allclose(elph_me[..., 138], 0.0)  # ediff should be zero for defect band
    assert np.linalg.norm(elph_me[..., 139]) > 0


def test_wswq_slope():
    # Make sure the the slope is automatically defined as the sign of the distoration changes.
    mats = [np.ones((3, 5)), np.zeros((3, 5)), np.ones((3, 5))]
    FakeWSWQ = namedtuple("FakeWSWQ", ["data"])
    fake_wswqs = [FakeWSWQ(data=m) for m in mats]

    res = _get_wswq_slope([-0.5, 0, 0.5], fake_wswqs)
    np.allclose(res, np.ones((3, 5)) * 2)

    res = _get_wswq_slope([1.0, 0, -1.0], fake_wswqs)
    np.allclose(res, np.ones((3, 5)) * -1)


def test_SRHCapture(hd0, hd1, test_dir):
    from pymatgen.analysis.defects.ccd import get_SRH_coefficient

    hd0.read_wswqs(test_dir / "v_Ga" / "ccd_0_-1" / "wswqs")
    c_n = get_SRH_coefficient(
        initial_state=hd0,
        final_state=hd1,
        defect_state=(138, 1, 1),
        T=[100, 200, 300],
        dE=1.0,
    )
    ref_results = [1.89187260e-34, 6.21019152e-33, 3.51501688e-31]
    assert np.allclose(c_n, ref_results)


def test_dielectric_func(test_dir):
    dir0_opt = test_dir / "v_Ga" / "ccd_0_-1" / "optics"
    hd0 = HarmonicDefect.from_directories(
        directories=[dir0_opt],
        store_bandstructure=True,
    )
    hd0.waveder = Waveder.from_binary(dir0_opt / "WAVEDER")
    energy, eps_vbm, eps_cbm = hd0.get_dielectric_function(idir=0, jdir=0)
    inter_vbm = np.trapz(np.imag(eps_vbm[:100]), energy[:100])
    inter_cbm = np.trapz(np.imag(eps_cbm[:100]), energy[:100])
    assert pytest.approx(inter_vbm, abs=0.01) == 6.31
    assert pytest.approx(inter_cbm, abs=0.01) == 0.27


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


# def test_SRHCapture(v_ga):
#     from pymatgen.analysis.defects.ccd import HarmonicDefect, SRHCapture

#     vaspruns = v_ga[(0, -1)]["vaspruns"]
#     procar = v_ga[(0, -1)]["procar"]
#     hd0 = HarmonicDefect.from_vaspruns(
#         vaspruns, charge_state=0, procar=procar, store_bandstructure=True
#     )

#     vaspruns = v_ga[(-1, 0)]["vaspruns"]
#     procar = v_ga[(-1, 0)]["procar"]
#     hdm1 = HarmonicDefect.from_vaspruns(
#         vaspruns, charge_state=-1, procar=procar, store_bandstructure=True
#     )
#     dQ = get_dQ(hd0.structures[hd0.relaxed_index], hdm1.structures[hdm1.relaxed_index])
#     srh_cap = SRHCapture(hd0, hdm1, dQ=dQ, wswqs=v_ga[(0, -1)]["wswqs"])

#     c_n = srh_cap.get_coeff(
#         T=[100, 200, 300],
#         dE=1.0,
#         volume=hd0.structures[hd0.relaxed_index].volume,
#         kpt_index=1,
#     )
#     ref_results = [1.89187260e-34, 6.21019152e-33, 3.51501688e-31]
#     assert np.allclose(c_n, ref_results)

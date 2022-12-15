import itertools

import numpy as np
import pytest

from pymatgen.analysis.defects.recombination import (
    analytic_overlap_NM,
    boltzmann_filling,
    get_mQn,
    get_Rad_coef,
    get_SRH_coef,
    pchip_eval,
)


def test_boltzmann():
    ref_results = [
        0.9791034813819097,
        0.020459854127734073,
        0.00042753972270360594,
        8.934091775449048e-06,
        1.8669141512139823e-07,
        3.901200631921917e-09,
    ]
    results = boltzmann_filling(0.1, 300, n_states=6)
    assert np.allclose(results.flatten(), ref_results, rtol=1e-3)
    results2 = boltzmann_filling(0.1, [100, 300], n_states=6)
    assert np.allclose(results2[:, 1], ref_results, rtol=1e-3)


def test_get_vibronic_matrix_elements():
    # precompute values of the overlap
    dQ, omega_i, omega_f = 0, 0.2, 0.2
    Ni, Nf = 5, 5
    ovl = np.zeros((Ni, Nf), dtype=np.longdouble)
    for m, n in itertools.product(range(Ni), range(Nf)):
        ovl[m, n] = analytic_overlap_NM(dQ, omega_i, omega_f, m, n)

    e, matel = get_mQn(
        omega_i=omega_i, omega_f=omega_f, m_init=0, Nf=Nf, dQ=dQ, ovl=ovl
    )
    ref_result = [0.0, 3984589.0407885523, 0.0, 0.0, 0.0]
    assert np.allclose(matel, ref_result)


def test_pchip_eval():
    x_c = np.linspace(0, 2, 5)
    y_c = np.sin(x_c) + 1
    xx = np.linspace(-3, 3, 1000)
    fx = pchip_eval(xx, x_coarse=x_c, y_coarse=y_c)
    int_val = np.trapz(np.nan_to_num(fx), x=xx)
    int_ref = np.sum(y_c)
    assert int_val == pytest.approx(int_ref, rel=1e-3)


def test_get_SRH_coef():
    ref_res = [4.64530153e-14, 4.64752885e-14, 4.75265302e-14]
    res = get_SRH_coef(
        T=[100, 200, 300],
        dQ=1.0,
        dE=1.0,
        omega_i=0.2,
        omega_f=0.2,
        elph_me=1,
        volume=1,
        g=1,
    )
    assert np.allclose(res, ref_res)


def test_get_Rad_coef():
    res = get_Rad_coef(
        T=[100, 200, 300],
        dQ=1.0,
        dE=1.0,
        omega_i=0.2,
        omega_f=0.2,
        omega_photon=0.6,
        dipole_me=1,
        volume=1,
        g=1,
    )

"""Recombination algorithms pure function implementations.

The the SRH recombination code is taken from the NonRad code (www.github.com/mturiansky/nonrad)
"""

from __future__ import annotations

import itertools

import numpy as np
from numba import njit
from numpy import typing as npt
from scipy.interpolate import PchipInterpolator

from .constants import AMU2KG, ANGS2M, EV2J, HBAR_EV, HBAR_J, KB, LOOKUP_TABLE

__author__ = "Jimmy Shen"
__copyright__ = "The Materials Project"
__maintainer__ = "Jimmy Shen"
__email__ = "jmmshn@gmail.com"
__date__ = "Mar 15, 2022"

Factor1 = ANGS2M**2 * AMU2KG / HBAR_EV / HBAR_EV / EV2J
Factor2 = HBAR_J / ANGS2M**2 / AMU2KG
Factor3 = 1 / HBAR_EV


@njit(cache=True)
def fact(n: int) -> float:  # pragma: no cover
    """Compute the factorial of n."""
    if n > 20:
        return LOOKUP_TABLE[-1] * np.prod(
            np.array(list(range(21, n + 1)), dtype=np.double)
        )
    return LOOKUP_TABLE[n]


@njit(cache=True)
def herm(x: float, n: int) -> float:  # pragma: no cover
    """Recursive definition of hermite polynomial."""
    if n == 0:
        return 1.0
    if n == 1:
        return 2.0 * x

    y1 = 2.0 * x
    dy1 = 2.0
    for i in range(2, n + 1):
        yn = 2.0 * x * y1 - dy1
        dyn = 2.0 * i * y1
        y1 = yn
        dy1 = dyn
    return yn


@njit(cache=True)
def analytic_overlap_NM(
    dQ: float, omega1: float, omega2: float, n1: int, n2: int
) -> float:  # pragma: no cover
    """Compute the overlap between two displaced harmonic oscillators.

    This function computes the overlap integral between two harmonic
    oscillators with frequencies w1, w2 that are displaced by DQ for the
    quantum numbers n1, n2. The integral is computed using an analytic formula
    for the overlap of two displaced harmonic oscillators. The method comes
    from B.P. Zapol, Chem. Phys. Lett. 93, 549 (1982).

    [Taken from NONRAD.]

    Args:
        DQ: Displacement between harmonic oscillators in amu^{1/2} Angstrom
        omega1: Frequency of oscillator 1 in eV
        omega2: Frequency of oscillator 2 in eV
        n1: Quantum number of oscillator 1
        n2: Quantum number of oscillator 2

    Returns:
        Overlap of the two harmonic oscillator wavefunctions
    """
    w = np.double(omega1 * omega2 / (omega1 + omega2))
    rho = np.sqrt(Factor1) * np.sqrt(w / 2) * dQ
    sinfi = np.sqrt(omega1) / np.sqrt(omega1 + omega2)
    cosfi = np.sqrt(omega2) / np.sqrt(omega1 + omega2)

    Pr1 = (-1) ** n1 * np.sqrt(2 * cosfi * sinfi) * np.exp(-(rho**2))
    Ix = 0.0
    k1 = n2 // 2
    k2 = n2 % 2
    l1 = n1 // 2
    l2 = n1 % 2
    for kx in range(k1 + 1):
        for lx in range(l1 + 1):
            k = 2 * kx + k2
            l = 2 * lx + l2  # noqa: E741
            Pr2 = (
                (fact(n1) * fact(n2)) ** 0.5
                / (fact(k) * fact(l) * fact(k1 - kx) * fact(l1 - lx))
                * 2 ** ((k + l - n2 - n1) / 2)
            )
            Pr3 = (sinfi**k) * (cosfi**l)
            f = herm(np.float64(rho), k + l)
            Ix = Ix + Pr1 * Pr2 * Pr3 * f
    return Ix


def boltzmann_filling(
    omega_i: float,
    temperature: npt.ArrayLike,
    n_states: int = 30,
) -> npt.NDArray:
    """Calculate the Boltzman filling of the phonon states.

    Get the Boltzmann filling of the lowest ``n_states`` at the given temperature(s).

    Args:
        omega_i: The phonon frequency in eV.
        temperature: The temperature in Kelvin.
        n_states: The number of states to consider.

    Returns:
        The Boltzmann filling factor as a matrix with
        shape ``(nstates, # temperature steps)``.
    """
    t_ = np.atleast_1d(temperature)
    Z = 1.0 / (1 - np.exp(-omega_i / KB / t_))
    m_omega = np.arange(0, n_states) * omega_i
    w = np.exp(-np.outer(m_omega, 1.0 / (KB * t_)))
    return np.multiply(w, 1 / Z)


def get_mQn(
    dQ: float,
    omega_i: float,
    omega_f: float,
    m_init: int,
    Nf: int,
    ovl: npt.NDArray,
):
    """Get the matrix element values for the position operator.

        <m_i|Q|n_f>

    Args:
        dQ: The displacement between the initial and final phonon states.
        omega_i: The initial phonon frequency in eV.
        omega_f: The final phonon frequency in eV.
        m_init: The initial phonon quantum number.
        Nf: The number of final phonon states.
        ovl: The overlap between the initial and final phonon states.

    Returns:
        np.array: The energy different different between energy states.
            This can be off-set by a constant value depending on the physical process you are studying.
        np.array: The matrix elements for those pairs of states.
    """
    E = np.linspace(0, Nf * omega_f, Nf, endpoint=False) - m_init * omega_i
    if m_init == 0:
        matels = (
            np.sqrt(Factor2 / 2 / omega_i) * ovl[m_init + 1, :]
            + np.sqrt(Factor3) * dQ * ovl[m_init, :]
        )
    else:
        matels = (
            np.sqrt((m_init + 1) * Factor2 / 2 / omega_i) * ovl[m_init + 1, :]
            + np.sqrt(m_init * Factor2 / 2 / omega_i) * ovl[m_init - 1, :]
            + np.sqrt(Factor3) * dQ * ovl[m_init, :]
        )
    return E, matels


def get_mn(
    dQ: float,
    omega_i: float,
    omega_f: float,
    m_init: int,
    en_final: float,
    en_pad: float = 0.5,
):
    """Get the matrix element values for the position operator.

        <m_i|n_f>
    Starting in state m_init and ending and ending on final states between
    ``en_final - en_pad`` and ``en_final + en_pad`` reference to the bottom of the final state parabola.

    Args:
        omega_i: The initial phonon frequency in eV.
        omega_f: The final phonon frequency in eV.
        m_init: The initial phonon quantum number.
        en_final: The final energy in eV, reference to the bottom of the
            final state parabola.
        en_pad: The energy window to consider in eV.

    Returns:
        np.array: The energy different different between energy states.
            This can be off-set by a constant value depending on the physical process you are studying.
        np.array: The matrix elements for those pairs of states.
    """
    n_min = max(int((en_final - en_pad) // omega_f), 0)
    n_max = int((en_final + en_pad) // omega_f) + 2
    E = np.arange(n_min, n_max) * omega_f
    matels = np.zeros_like(E)
    for n in range(n_min, n_max):
        matels[n - n_min] = analytic_overlap_NM(
            dQ=dQ, omega1=omega_i, omega2=omega_f, n1=m_init, n2=n
        )
    return E, matels


def pchip_eval(
    x: npt.ArrayLike | float,
    x_coarse: npt.ArrayLike,
    y_coarse: npt.ArrayLike,
    pad_frac: float = 0.2,
    n_points: int = 5000,
):
    """Evaluate a piecewise cubic Hermite interpolant.

    Assuming a function is evenly sampleded on (``x_coarse``, ``y_coarse``),
    Then we know the final shape of the function has to satisfy match the coarsely sampled data.
    Thus, we can just interpolate the function on a finer grid and ensure that the interpolated function
    Integrates to the same value as the sum of the coarsely sampled data.

    Args:
        x: The x value/values to evaluate the interpolant at.
        x_coarse: The x values of the coarsely sampled data.
        y_coarse: The y values of the coarsely sampled data.
        pad_frac: The fraction of the domain to pad the interpolation by.
        n_points: The number of points to evaluate the interpolant at.

    Returns:
        The interpolated values. Note that if ``x`` or any of the elements of ``x`` are outside the domain,
        then the returned value will be ``np.nan``.

    """
    x_min, x_max = min(x_coarse), max(x_coarse)
    x_pad = abs(x_max - x_min) * pad_frac
    interp_domain = np.linspace(x_min - x_pad, x_max + x_pad, n_points)
    interp_func = PchipInterpolator(x_coarse, y_coarse, extrapolate=False)
    return (
        interp_func(x)
        * np.sum(y_coarse)
        / np.trapz(np.nan_to_num(interp_func(interp_domain)), x=interp_domain)
    )


def get_SRH_coef(
    T: float | npt.ArrayLike,
    dQ: float,
    dE: float,
    omega_i: float,
    omega_f: float,
    elph_me: float,
    volume: float,
    g: int = 1,
    occ_tol: float = 1e-3,
) -> npt.ArrayLike:
    """Compute the SRH recombination Coefficient.

    Args:
        T: The temperature in Kelvin.
        dQ: The displacement between the initial and final phonon states. In units of amu^{1/2} Angstrom.
        dE: The energy difference between the initial and final phonon states. In units of eV.
        omega_i: The initial phonon frequency in eV.
        omega_f: The final phonon frequency in eV.
        elph_me: The electron-phonon matrix element in units of eV amu^{-1/2} Angstrom^{-1}
        volume: The volume of the simulation cell in Angstrom^3.
        g: The degeneracy factor of the final state.
        occ_tol : Ni is chosen so that (1 - occ_tol) of the total Bose-Einstein occupation is included.

    Returns:
        Resulting capture coefficient (unscaled) in cm^3 s^{-1}
    """
    volume *= (1e-8) ** 3  # Convert to cm^3
    T = np.atleast_1d(T)
    kT = KB * max(T)
    Ni, Nf = (17, 50)
    tNi = np.ceil(-np.max(kT) * np.log(occ_tol) / omega_i).astype(int)
    Ni = max(Ni, tNi)
    tNf = np.ceil((dE + Ni * omega_i) / omega_f).astype(int)
    Nf = max(Nf, tNf)
    Ni = Ni - 1

    ovl = np.zeros((Ni + 1, Nf), dtype=np.longdouble)
    for m, n in itertools.product(range(Ni + 1), range(Nf)):
        ovl[m, n] = analytic_overlap_NM(dQ, omega_i, omega_f, m, n)
    weights = boltzmann_filling(omega_i, T, Ni)
    rate = np.zeros_like(T, dtype=np.longdouble)
    for m in range(Ni):
        E, me = get_mQn(
            dQ=dQ, omega_i=omega_i, omega_f=omega_f, m_init=m, Nf=Nf, ovl=ovl
        )
        interp_me = pchip_eval(
            dE, E, np.abs(np.conj(me) * me), pad_frac=0.2, n_points=5000
        )
        rate += weights[m, :] * interp_me
    return 2 * np.pi * g * elph_me**2 * volume * rate


def get_Rad_coef(
    T: float | npt.ArrayLike,
    dQ: float,
    dE: float,
    omega_i: float,
    omega_f: float,
    omega_photon: float,
    dipole_me: float,
    volume: float,
    g: int = 1,
    occ_tol: float = 1e-3,
) -> npt.ArrayLike:
    """Compute the Radiative recombination Coefficient.

    We assumed that the transition takes each initial state to some final state.
    We still use the interpolation method in place of numerical smearing, although
    that might change if we start to consider broader photon frequencies.

    Args:
        T: The temperature in Kelvin.
        dQ: The displacement between the initial and final phonon states. In units of amu^{1/2} Angstrom.
        dE: The energy difference between the initial and final phonon states. In units of eV.
        omega_i: The initial phonon frequency in eV.
        omega_f: The final phonon frequency in eV.
        omega_photon: The photon frequency in eV.
        dipole_me: The dipole matrix element in units of eV amu^{-1/2} Angstrom^{-1}
        volume: The volume of the simulation cell in Angstrom^3.
        g: The degeneracy factor of the final state.
        occ_tol : Ni is chosen so that (1 - occ_tol) of the total Bose-Einstein weight is included.

    Returns:
        Resulting capture coefficient (unscaled) in cm^3 s^{-1}
    """
    volume *= (1e-8) ** 3  # Convert to cm^3
    T = np.atleast_1d(T)
    kT = KB * max(T)
    Ni = max(17, int(np.ceil(-np.max(kT) * np.log(occ_tol) / omega_i)))
    weights = boltzmann_filling(omega_i, T, Ni)
    rate = np.zeros_like(T, dtype=np.longdouble)

    for m in range(Ni):
        final_energy = (m * omega_i) + dE - omega_photon
        E, me = get_mn(
            dQ=dQ,
            omega_i=omega_i,
            omega_f=omega_f,
            m_init=m,
            en_final=final_energy,
        )
        if len(E) <= 1:
            # The photon took you to a energy range with no final states
            continue
        interp_me = pchip_eval(final_energy, E, me * me, pad_frac=0.2, n_points=5000)
        rate += weights[m, :] * interp_me

    return 2 * np.pi * g * dipole_me**2 * volume * rate

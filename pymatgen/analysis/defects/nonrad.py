"""Re-implementation of the Nonrad code with more vectorization."""

import numpy as np
from numba import njit
from numpy import typing as npt
from scipy.interpolate import PchipInterpolator

from .constants import AMU2KG, ANGS2M, EV2J, HBAR, HBAR_EV, KB, LOOKUP_TABLE

Factor1 = ANGS2M**2 * AMU2KG / HBAR_EV / HBAR_EV / EV2J
Factor2 = HBAR / ANGS2M**2 / AMU2KG
Factor3 = 1 / HBAR_EV


@njit(cache=True)
def fact(n: int) -> float:
    """Compute the factorial of n."""
    if n > 20:
        return LOOKUP_TABLE[-1] * np.prod(
            np.array(list(range(21, n + 1)), dtype=np.double)
        )
    return LOOKUP_TABLE[n]


@njit(cache=True)
def herm(x: float, n: int) -> float:
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
) -> float:
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
            # f = hermval(rho, [0.]*(k+l) + [1.])
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


def get_vibronic_matrix_elements(
    omega_i: float,
    omega_f: float,
    m_init: int,
    Nf: int,
    dQ: float,
    ovl: npt.NDArray,
):
    """Get the vibronic matrix element values.

    Args:
        omega_i: The initial phonon frequency in eV.
        omega_f: The final phonon frequency in eV.
        m_init: The initial phonon quantum number.
        Nf: The number of final phonon states.
        dQ: The displacement between the initial and final phonon states.
        ovl: The overlap between the initial and final phonon states.

    Returns:
        np.array: The energy different different between energy states.
            This can be off-set by a constant value depending on the physical process you are studying.
        np.array: The matrix elements for those pairs of states.
    """
    E = np.arange(0, Nf, omega_f) - m_init * omega_i
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
        The interpolated values.

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

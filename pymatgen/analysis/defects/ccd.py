"""Configuration-coordinate diagram analysis."""
from __future__ import annotations

import logging
from ctypes import Structure
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import numpy.typing as npt
from monty.json import MSONable
from numba import njit
from pymatgen.electronic_structure.core import Spin
from pymatgen.io.vasp.outputs import WSWQ, BandStructure, Procar, Vasprun, Waveder
from scipy import constants as const
from scipy.optimize import curve_fit

from .utils import get_localized_state, sort_positive_definite

# Copyright (c) Pymatgen Development Team.
# Distributed under the terms of the MIT License.

__author__ = "Jimmy Shen"
__copyright__ = "The Materials Project"
__maintainer__ = "Jimmy Shen"
__email__ = "jmmshn@gmail.com"
__date__ = "Mar 15, 2022"
_logger = logging.getLogger(__name__)

HBAR = const.hbar / const.e  # in units of eV.s
EV2J = const.e  # 1 eV in Joules
AMU2KG = const.physical_constants["atomic mass constant"][0]
ANGS2M = 1e-10  # angstrom in meters
KB = const.k / const.e  # Boltzmann constant in eV/K

AU2ANG = const.physical_constants["atomic unit of length"][0] / 1e-10
RYD2EV = const.physical_constants["Rydberg constant times hc in eV"][0]
EDEPS = 4 * np.pi * 2 * RYD2EV * AU2ANG  # exactly the same as VASP


def optical_prefactor(struct):
    """Prefactor for optical transition rate calculations."""
    return EDEPS * np.pi / struct.volume


# __all__ = ["ConfigurationCoordinateDiagram", "HarmonicDefect", "get_dQ"]


@dataclass
class HarmonicDefect(MSONable):
    """A class representing the a harmonic defect vibronic state.

    The vibronic part of a defect is often catured by a simple harmonic oscillator.
    This class store a representation of the SHO as well as some additional information for book-keeping purposes.

    Attributes:
        omega: The vibronic frequency of the phonon state in in the same units as the energy vs. Q plot.
        charge: The charge state. This should be the charge of the defect
            simulation that gave rise to the minimum of the parabola.
        distortions: The distortion of the structure in units of amu^{-1/2} Angstrom^{-1}.
            This object's internal reference for the distoration should always be relaxed structure.
        structures: The list of structures that were used to compute the distortions.
        energies: The potential energy surface obtained by distorting the structure.A
        defect_band_index: The index of the defect band.
        relaxed_indices: The indices of the relaxed defect structure.
        relaxed_bandstructure: The band structure of the relaxed defect calculation.
    """

    omega: float
    charge_state: int
    distortions: Optional[list[float]] = None
    structures: Optional[list[Structure]] = None
    energies: Optional[list[float]] = None
    defect_band_index: Optional[int] = None
    relaxed_index: Optional[int] = None
    relaxed_bandstructure: Optional[BandStructure] = None

    @classmethod
    def from_vaspruns(
        cls,
        vasp_runs: list[Vasprun],
        charge_state: int,
        relaxed_index: int | None = None,
        defect_band_index: int | None = None,
        procar: Procar | None = None,
        store_bandstructure: bool = False,
        get_band_structure_kwargs: dict | None = None,
        **kwargs,
    ) -> HarmonicDefect:
        """Create a HarmonicDefectPhonon from a list of vasprun.

        .. note::
            The constructor check that you have the vaspruns sorted by the distortions
            but does not perform the sorting for you.  This serves as a safety check to
            ensure that the vaspruns are properly ordered.

        Args:
            vasp_runs: A list of Vasprun objects.
            charge_state: The charge state for the defect.
            relaxed_index: The index of the relaxed structure in the list of structures.
            defect_band_index: The index of the defect band (0-indexed).
            procar: A Procar object.  Used to identify the defect band if the defect_band_index is not provided.
            store_bandstructure: Whether to store the bandstructure of the relaxed defect calculation.
                Defaults to False to save space.
            get_band_structure_kwargs: Keyword arguments to pass to the ``get_band_structure`` method.
            **kwargs: Additional keyword arguments to pass to the constructor.

        Returns:
            A HarmonicDefect object.
        """

        def _parse_vasprun(vasprun: Vasprun):
            energy = vasprun.final_energy
            struct = vasprun.final_structure
            return (energy, struct)

        energy_struct = list(map(_parse_vasprun, vasp_runs))
        unsorted_e = [e for e, _ in energy_struct]

        if relaxed_index is None:
            # Use the vasprun with the lowest energy
            relaxed_index = np.argmin([e for e, _ in energy_struct])

        sorted_list, distortions = sort_positive_definite(
            energy_struct,
            energy_struct[relaxed_index],
            energy_struct[-1],
            lambda x, y: get_dQ(x[1], y[1]),
        )
        energies, structures = list(zip(*sorted_list))

        if not np.allclose(unsorted_e, energies, atol=1e-99):
            raise ValueError("The vaspruns should already be in order.")

        omega = _get_omega(
            Q=distortions,
            E=energies,
            Q0=distortions[relaxed_index],
            E0=energies[relaxed_index],
        )

        if defect_band_index is None and procar is not None:
            bandstructure = vasp_runs[relaxed_index].get_band_structure()
            loc_res = get_localized_state(bandstructure=bandstructure, procar=procar)
            _, (_, defect_band_index) = min(loc_res.items(), key=lambda x: x[1])

        if store_bandstructure:
            get_band_structure_kwargs = get_band_structure_kwargs or {}
            bs = vasp_runs[relaxed_index].get_band_structure(
                **get_band_structure_kwargs
            )
        else:
            bs = None

        return cls(
            omega=omega,
            charge_state=charge_state,
            structures=structures,
            distortions=distortions,
            energies=energies,
            defect_band_index=defect_band_index,
            relaxed_index=relaxed_index,
            relaxed_bandstructure=bs,
            **kwargs,
        )

    @property
    def omega_eV(self) -> float:
        """Get the vibronic frequency of the phonon state in (eV)."""
        return self.omega * HBAR * np.sqrt(EV2J / (ANGS2M**2 * AMU2KG))

    def occupation(self, t: npt.ArrayLike | float) -> npt.ArrayLike:
        """Calculate the phonon occupation.

        Args:
            t: The temperature in Kelvin.
        """
        return 1.0 / (1 - np.exp(-self.omega_eV / KB * t))

    def get_elph_me(
        self,
        wswqs: list[WSWQ],
    ) -> npt.NDArray:
        """Calculate the electron phonon matrix elements.

        Combine the data from the WSWQs to calculate the electron phonon matrix elements.
        The matrix elements are calculated by combining the finite difference from the matrix overlaps.

        d(<W|S|W(Q)>) / dQ

        And the eignvalue difference.

        Args:
            wswqs: A list of WSWQ objects, assuming that they match the order of the distortions.
            bandstructure: The bandstructure of the relaxed defect calculation.

        Returns:
            npt.NDArray: The electron phonon matrix elements.
        """
        if self.defect_band_index is None:
            raise ValueError("The ``defect_band_index`` must be already be set.")

        # It's either [..., defect_band_index, :] or [..., defect_band_index]
        # Which band index is the "correct" one might not be super important since
        # the matrix is symmetric in the first-order theory we are working in.
        # TODO: I should really read my thesis.
        slopes = _get_wswq_slope(self.distortions, wswqs)[
            ..., self.defect_band_index, :
        ]
        ediffs = self._get_ediff(output_order="skb")
        return np.multiply(slopes, ediffs)

    def _get_ediff(self, output_order="skb") -> npt.NDArray:
        """Compute the eigenvalue difference to the defect band.

        .. note::
            Since the different matrix element output files have different index orders,
            But most calculations require the energies, we should always perform the
            rearrangement here so that we have a single point of failure.

        Args:
            band_structure: The band structure of the relaxed defect calculation.
            output_order: The order of the output. Defaults to "skb" (spin, kpoint, band]).
                You can also use "bks" (band, kpoint, spin).


        Returns:
            The eigenvalue difference to the defect band in the order specified by output_order.
        """
        if self.defect_band_index is None:
            raise ValueError(  # pragma: no cover
                "The ``defect_band_index`` must be set before ``ediff`` can be computed."
            )
        if self.relaxed_bandstructure is None:
            raise ValueError(  # pragma: no cover
                "The ``relaxed_bandstructure`` must be set before ``ediff`` can be computed."
            )

        ediffs_ = _get_ks_ediff(
            bandstructure=self.relaxed_bandstructure,
            defect_band_index=self.defect_band_index,
        )
        ediffs_stack = [
            ediffs_[Spin.up].T,
        ]
        if Spin.down in ediffs_.keys():
            ediffs_stack.append(ediffs_[Spin.down].T)
        ediffs = np.stack(ediffs_stack)

        if output_order == "skb":
            return ediffs
        elif output_order == "bks":
            return ediffs.transpose((2, 1, 0))
        else:
            raise ValueError(
                "Invalid output_order, choose from 'skb' or 'bks'."
            )  # pragma: no cover


@dataclass
class OpticalHarmonicDefect(HarmonicDefect):
    """Representation of Harmonic defect with optical (dipole) matrix elements.

    The dipole matrix elements are computed by VASP and reported in the WAVEDER file.

    Attributes:
        omega: The vibronic frequency of the phonon state in in the same units as the energy vs. Q plot.
        charge: The charge state. This should be the charge of the defect
            simulation that gave rise to the minimum of the parabola.
        distortions: The distortion of the structure in units of amu^{-1/2} Angstrom^{-1}.
            This object's internal reference for the distoration should always be relaxed structure.
        structures: The list of structures that were used to compute the distortions.
        energies: The potential energy surface obtained by distorting the structure.A
        defect_band_index: The index of the defect band.
        relaxed_indices: The indices of the relaxed defect structure.
        relaxed_bandstructure: The band structure of the relaxed defect calculation.
        waveder: The WAVEDER object containing the dipole matrix elements.
    """

    # TODO: use kw_only once we drop Python < 3.10
    waveder: Waveder | None = None

    @classmethod
    def from_vaspruns_and_waveder(
        cls,
        vasp_runs: list[Vasprun],
        waveder: Waveder,
        charge_state: int,
        relaxed_index: int | None = None,
        defect_band_index: int | None = None,
        procar: Procar | None = None,
        get_band_structure_kwargs: dict | None = None,
        **kwargs,
    ) -> OpticalHarmonicDefect:
        """Create a HarmonicDefectPhonon from a list of vasprun.

        .. note::
            The constructor check that you have the vaspruns sorted by the distortions
            but does not order it for you.

        Args:
            vasp_runs: A list of Vasprun objects.
            charge_state: The charge state for the defect.
            relaxed_index: The index of the relaxed structure in the list of structures.
            defect_band_index: The index of the defect band (0-indexed).
            procar: The Procar object for the defect calculation.

        Returns:
            An OpticalHarmonicDefect object.
        """
        obj = super().from_vaspruns(
            vasp_runs,
            charge_state,
            relaxed_index,
            waveder=waveder,
            defect_band_index=defect_band_index,
            procar=procar,
            store_bandstructure=True,
            get_band_structure_kwargs=get_band_structure_kwargs,
            **kwargs,
        )
        if obj.defect_band_index is None:
            raise ValueError(  # pragma: no cover
                "You must provide `defect_band_index` or PROCAR to help indetify the `defect_band_index`."
            )
        if obj.relaxed_bandstructure is None:
            raise ValueError(  # pragma: no cover
                "The bandstructure was not populated properly check the constructor of the parent."
            )
        return obj

    @classmethod
    def from_vaspruns(
        cls,
        vasp_runs: list[Vasprun],
        charge_state: int,
        relaxed_index: int | None = None,
        defect_band_index: int | None = None,
        procar: Procar | None = None,
        store_bandstructure: bool = False,
        get_band_structure_kwargs: dict | None = None,
        **kwargs,
    ) -> HarmonicDefect:
        """Not implemented."""
        raise NotImplementedError("Use from_vaspruns_and_waveder instead.")

    def _get_defect_dipoles(self) -> npt.NDArray:
        """Get the dipole matrix elements for the defect.

        Returns:
            The dipole matrix elements for the defect. The indices are:
                ``[band index, k-point index, spin index, cart. direction]``.
        """
        return self.waveder.cder_data[self.defect_band_index, ...]

    def _get_spectra(self) -> npt.NDArray:
        """Get the spectra for the defect.

        Args:
            bandstructure: The band structure of the relaxed defect calculation.
            shift: The shift to apply to the spectra.
        """
        return self._get_ediff(output_order="bks")


# @dataclass
# class ConfigurationCoordinateDiagram(MSONable):
#     """A class representing a configuration coordinate diagram.

#     The configuration coordinate diagram represents two parabolas with some finite configuration shift ``dQ``.
#     The two defects are ``sorted`` in the sense that the defect with the lower ``charge_state``
#       is designated as ``defect_state_0``.

#     Attributes:
#         phonon_mode_0 : The defect with the lower charge state.
#         phonon_mode_1 : The defect with the higher charge state.
#         dQ : The finite configuration shift.
#     """
#     phonon_mode_0: HarmonicDefect
#     phonon_mode_1: HarmonicDefect
#     dQ: float

#     def __post_init__(self):
#         """Post-initialization."""
#         if abs(self.phonon_mode_0.charge_state - self.phonon_mode_1.charge_state) != 1:
#             raise ValueError(
#                 "The charge states of the two defects must be 1 apart. "
#                 "Got {} and {}".format(self.phonon_mode_0.charge_state, self.phonon_mode_1.charge_state)
#             )
#         if self.phonon_mode_0.charge_state > self.phonon_mode_1.charge_state:
#             self.phonon_mode_0, self.phonon_mode_1 = self.phonon_mode_1, self.phonon_mode_0

#     @property
#     def omega0_eV(self) -> float:
#         """The vibronic frequency of the defect with the lower charge state."""
#         return self.phonon_mode_0.omega_eV

#     @property
#     def omega1_eV(self) -> float:
#         """The vibronic frequency of the defect with the higher charge state."""
#         return self.phonon_mode_1.omega_eV


def get_dQ(ground: Structure, excited: Structure) -> float:
    """Calculate configuration coordinate difference.

    Args:
        ground : pymatgen structure corresponding to the ground (final) state
        excited : pymatgen structure corresponding to the excited (initial) state

    Returns:
        (float):  the dQ value (amu^{1/2} Angstrom)
    """
    return np.sqrt(
        np.sum(
            list(
                map(
                    lambda x: x[0].distance(x[1]) ** 2 * x[0].specie.atomic_mass,
                    zip(ground, excited),
                )
            )
        )
    )


def _get_omega(
    Q: npt.ArrayLike,
    E: npt.ArrayLike,
    Q0: float,
    E0: float,
) -> float:
    """Calculate the omega from the PES.

    Taken from NONRAD

    Args:
        Q: array of Q values (amu^{1/2} Angstrom) corresponding to each vasprun
        E: array of energy values (eV) corresponding to each vasprun
        Q0: fix the x-value of the minimum of the parabola
        E0: fix the y-value of the minimum of the parabola

    Returns:
        omega: the harmonic phonon frequency in (eV)
    """
    popt = _fit_parabola(Q, E, Q0, E0)
    return popt[0]


def _fit_parabola(
    Q: npt.ArrayLike, energy: npt.ArrayLike, Q0: float, E0: float
) -> Tuple[float, float, float]:
    """Fit the parabola to the data."""

    def f(Q, omega):
        """Get the parabola function."""
        return 0.5 * omega**2 * (Q - Q0) ** 2 + E0

    popt, _ = curve_fit(f, Q, energy)
    return popt


def _get_wswq_slope(distortions: list[float], wswqs: list[WSWQ]) -> npt.NDArray:
    """Get the slopes of the overlap matrixs vs. Q.

    Args:
        distortions: List of Q values (amu^{1/2} Angstrom).
        wswqs: List of WSWQ objects.

    Returns:
        npt.NDArray: slope matrix with the same shape as the ``WSWQ.data``.
    """
    yy = np.stack([np.abs(ww.data) * np.sign(qq) for qq, ww in zip(distortions, wswqs)])
    _, *oldshape = yy.shape
    return np.polyfit(distortions, yy.reshape(yy.shape[0], -1), deg=1)[0].reshape(
        *oldshape
    )


def _get_ks_ediff(
    bandstructure: BandStructure,
    defect_band_index: int,
) -> dict[Spin, npt.NDArray]:
    """Calculate the Kohn-Sham energy between the defect state.

    Get the eigenvalue differences to the defect band. Report this difference
    on each k-point and each spin, the result should be shape [nspins, nkpoints, nbands].

    Args:
        bandstructure: A BandStructure object.
        defect_band_index: The index of the defect band.

    Returns:
        npt.NDArray: The Kohn-Sham energy difference between the defect state and other states.
        Indexed the same way as ``bandstructure.bands``.
    """
    res = dict()
    for k, kpt_bands in bandstructure.bands.items():
        e_at_def_band = kpt_bands[defect_band_index, :]
        e_diff = kpt_bands - e_at_def_band
        res[k] = e_diff
    return res


HBAR = const.hbar / const.e  # in units of eV.s
EV2J = const.e  # 1 eV in Joules
AMU2KG = const.physical_constants["atomic mass constant"][0]
ANGS2M = 1e-10  # angstrom in meters

LOOKUP_TABLE = np.array(
    [
        1,
        1,
        2,
        6,
        24,
        120,
        720,
        5040,
        40320,
        362880,
        3628800,
        39916800,
        479001600,
        6227020800,
        87178291200,
        1307674368000,
        20922789888000,
        355687428096000,
        6402373705728000,
        121645100408832000,
        2432902008176640000,
    ],
    dtype=np.double,
)

factor = ANGS2M**2 * AMU2KG / HBAR / HBAR / EV2J


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
def analytic_overlap_NM(DQ: float, w1: float, w2: float, n1: int, n2: int) -> float:
    """Compute the overlap between two displaced harmonic oscillators.

    This function computes the overlap integral between two harmonic
    oscillators with frequencies w1, w2 that are displaced by DQ for the
    quantum numbers n1, n2. The integral is computed using an analytic formula
    for the overlap of two displaced harmonic oscillators. The method comes
    from B.P. Zapol, Chem. Phys. Lett. 93, 549 (1982).

    Taken from NONRAD.

    Parameters
    ----------
    DQ : float
        displacement between harmonic oscillators in amu^{1/2} Angstrom
    w1, w2 : float
        frequencies of the harmonic oscillators in eV
    n1, n2 : integer
        quantum number of the overlap integral to calculate
    Returns
    -------
    np.longdouble
        overlap of the two harmonic oscillator wavefunctions
    """
    w = np.double(w1 * w2 / (w1 + w2))
    rho = np.sqrt(factor) * np.sqrt(w / 2) * DQ
    sinfi = np.sqrt(w1) / np.sqrt(w1 + w2)
    cosfi = np.sqrt(w2) / np.sqrt(w1 + w2)

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

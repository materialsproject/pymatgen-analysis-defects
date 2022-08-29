"""Configuration-coordinate diagram analysis."""
from __future__ import annotations

import logging
from ctypes import Structure
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from monty.json import MSONable
from numpy.typing import ArrayLike, NDArray
from pymatgen.core import Spin
from pymatgen.io.vasp.outputs import WSWQ, BandStructure, Procar, Vasprun
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
__logger = logging.getLogger(__name__)

HBAR = const.hbar / const.e  # in units of eV.s
EV2J = const.e  # 1 eV in Joules
AMU2KG = const.physical_constants["atomic mass constant"][0]
ANGS2M = 1e-10  # angstrom in meters

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
        distortions : The distortion of the structure in units of amu^{-1/2} Angstrom^{-1}.
            This object's internal reference for the distoration should always be relaxed structure.
        energies : The potential energy surface obtained by distorting the structure.
    """

    omega: float
    charge_state: int
    structures: Optional[list[Structure]] = None
    distortions: Optional[list[float]] = None
    energies: Optional[list[float]] = None

    @classmethod
    def from_vaspruns(
        cls, vasp_runs: list[Vasprun], charge_state: int, relaxed_index: int | None = 0
    ) -> HarmonicDefect:
        """Create a HarmonicDefectPhonon from a list of vasprun.

        .. note::
            The constructor check that you have the vaspruns sorted by the distortions
            but does not order it for you.

        Args:
            vasp_runs: A list of Vasprun objects.
            charge_state: The charge state for the defect.
            relaxed_index: The index of the relaxed structure in the list of structures.

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
            return_eV=False,
        )

        return cls(
            omega=omega,
            charge_state=charge_state,
            structures=structures,
            distortions=distortions,
            energies=energies,
        )

    @property
    def omega_eV(self) -> float:
        """The vibronic frequency of the phonon state in (eV)."""
        return self.omega * HBAR * np.sqrt(EV2J / (ANGS2M**2 * AMU2KG))

    def get_elph_me(
        self,
        wswqs: list[WSWQ],
        defect_band_index: tuple[Spin, int, int] | None = None,
        procar: Procar | None = None,
    ):
        """Calculate the electron phonon matrix elements.

        Combine the data from the WSWQs to calculate the electron phonon matrix elements.
        The matrix elements are calculated by combining the finite difference from the matrix overlaps.

        d(<W|S|W(Q)>) / dQ

        And the eignvalue difference.

        Args:
            wswqs: A list of WSWQ objects, assuming that they match the order of the distortions.
            defect_band_index: The index of the defect band.
            procar: A Procar object.

        """
        _get_wswq_slope(self.distortions, wswqs)


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
    Q: ArrayLike,
    E: ArrayLike,
    Q0: float,
    E0: float,
    return_eV: bool = False,
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
    if return_eV:
        return HBAR * np.sqrt(EV2J / (ANGS2M**2 * AMU2KG)) * popt[0]
    else:
        return popt[0]


def _fit_parabola(
    Q: ArrayLike, energy: ArrayLike, Q0: float, E0: float
) -> Tuple[float, float, float]:
    """Fit the parabola to the data."""

    def f(Q, omega):
        """The parabola function."""
        return 0.5 * omega**2 * (Q - Q0) ** 2 + E0

    popt, _ = curve_fit(f, Q, energy)
    return popt


def _get_wswq_slope(distortions: list[float], wswqs: list[WSWQ]) -> NDArray:
    """Get the slopes of the overlap matrixs vs. Q.

    Args:
        distortions: List of Q values (amu^{1/2} Angstrom).
        wswqs: List of WSWQ objects.

    Returns:
        NDArray: slope matrix with the same shape as the ``WSWQ.data``.
    """
    yy = np.stack([np.abs(ww.data) * np.sign(qq) for qq, ww in zip(distortions, wswqs)])
    _, *oldshape = yy.shape
    return np.polyfit(distortions, yy.reshape(yy.shape[0], -1), deg=1)[0].reshape(
        *oldshape
    )


def _get_ks_ediff(
    delta_bands: list[int],
    bandstructure: BandStructure,
    procar: Procar,
    k_index: int,
    band_window: int = 5,
):
    """Calculate the Kohn-Sham energy between the defect state and a list of other states."""
    loc_states = get_localized_state(
        bandstructure, procar, k_index, band_window=band_window
    )
    res = dict()
    for spin, (ipr, min_band) in loc_states.items():
        def_eig = bandstructure.bands[spin][min_band, k_index]
        val_bands = np.array(delta_bands) + min_band
        val_eigs = bandstructure.bands[spin][val_bands, k_index]
        res[spin] = (ipr, dict(zip(val_bands, val_eigs - def_eig)))
    return res

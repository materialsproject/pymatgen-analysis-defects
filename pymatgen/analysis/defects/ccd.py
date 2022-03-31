from __future__ import annotations

# Copyright (c) Pymatgen Development Team.
# Distributed under the terms of the MIT License.

"""
Base classes representing defects.
"""


from ctypes import Structure
from monty.json import MSONable
from dataclasses import dataclass

import numpy as np

# from pymatgen.util.typing import ArrayLike
from scipy.optimize import curve_fit
from scipy import constants as const
from typing import Tuple
import logging
from nonrad.nonrad import HBAR, EV2J, AMU2KG, ANGS2M, get_C

__author__ = "Jimmy Shen"
__copyright__ = "The Materials Project"
__maintainer__ = "Jimmy Shen"
__email__ = "jmmshn@gmail.com"
__date__ = "Mar 15, 2022"
__logger = logging.getLogger(__name__)


@dataclass
class ConfigurationCoordinateDiagram(MSONable):
    """
    A class representing a configuration coordinate diagram.

    Based on the NONRAD code:
        M. E. Turiansky et al.: Comput. Phys. Commun. 267, 108056 (2021).

    Since configuration coordinate diagrams always represent some kind of a process in a defect.
    We will call one state `_gs` for ground state and another state `_es` for excited state.

    Since the CCD typically represents a interaction of a defect
    """

    charge_gs: int
    charge_es: int
    # distortions in units of [amu^{1/2} Angstrom]
    dQ: float
    Q_gs: np.array
    Q_es: np.array
    # energies in units of [eV]
    energies_gs: np.array
    energies_es: np.array
    # zero-phonon line energy in units of [eV]
    dE: float
    # electron-phonon matrix element Wif in units of
    # eV amu^{-1/2} Angstrom^{-1} for each bulk_index

    def __post_init__(self):
        self.Q_gs = np.array(self.Q_gs)
        self.Q_es = np.array(self.Q_es)
        self.energies_gs = np.array(self.energies_gs)
        self.energies_es = np.array(self.energies_es)

        # reference energies to zero:
        idx_zero = np.argmin(np.abs(self.Q_gs))
        idx_Q = np.argmin(np.abs(self.Q_es - self.dQ))
        self.energies_gs -= self.energies_gs[idx_zero]
        self.energies_es -= self.energies_es[idx_Q] - self.dE
        # get frequencies
        self.omega_gs = _get_omega(self.Q_gs, self.energies_gs, 0, 0)
        self.omega_es = _get_omega(self.Q_es, self.energies_es, self.dQ, self.dE)

    def fit_gs(self, Q):
        """
        Fit the ground state energy to a parabola.
        """
        E0 = 0
        omega = _fit_parabola(self.Q_gs, self.energies_gs, 0, E0)
        return 0.5 * omega**2 * (Q) ** 2 + E0

    def fit_es(self, Q):
        """
        Fit the excited state energy to a parabola.
        """
        E0 = self.dE
        omega = _fit_parabola(self.Q_es, self.energies_es, self.dQ, E0)
        return 0.5 * omega**2 * (Q - self.dQ) ** 2 + E0

    def plot(self, ax=None, show=True, **kwargs):
        """
        Plot the configuration coordinate diagram.
        """
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots()
        (l_gs,) = ax.plot(self.Q_gs, self.energies_gs, "o", label="gs", **kwargs)
        (l_es,) = ax.plot(self.Q_es, self.energies_es, "o", label="es", **kwargs)

        qq = np.linspace(self.Q_gs.min() - 0.2, self.Q_es.max() + 0.2, 100)
        ax.plot(qq, self.fit_gs(qq), "-", color=l_gs.get_color())
        ax.plot(qq, self.fit_es(qq), "-", color=l_es.get_color())

        ax.set_xlabel("Q [amu^{1/2} Angstrom]")
        ax.set_ylabel("Energy [eV]")
        ax.legend()
        if show:
            plt.show()
        return ax

def get_dQ(ground: Structure, excited: Structure) -> float:
    """Calculate dQ from the initial and final structures.
    Parameters
    ----------
    ground : pymatgen.core.structure.Structure
        pymatgen structure corresponding to the ground (final) state
    excited : pymatgen.core.structure.Structure
        pymatgen structure corresponding to the excited (initial) state
    Returns
    -------
    float
        the dQ value (amu^{1/2} Angstrom)
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
    Q: np.array,
    energy: np.array,
    Q0: float,
    E0: float,
) -> float:
    """
    Calculate the omega from the PES.

    Taken from NONRAD

    Args:
        Q: array of Q values (amu^{1/2} Angstrom) corresponding to each vasprun
        energy: array of energy values (eV) corresponding to each vasprun
        Q0: fix the value of the minimum of the parabola

    Returns:
        omega: the harmonic phonon frequency in (eV)
    """

    popt = _fit_parabola(Q, energy, Q0, E0)
    return HBAR * popt[0] * np.sqrt(EV2J / (ANGS2M**2 * AMU2KG))

def _fit_parabola(
    Q: np.array, energy: np.array, Q0: float, E0: float
) -> Tuple[float, float, float]:
    """
    Fit the parabola to the data.
    """

    def f(Q, omega):
        return 0.5 * omega**2 * (Q - Q0) ** 2 + E0

    popt, _ = curve_fit(f, Q, energy)
    return popt

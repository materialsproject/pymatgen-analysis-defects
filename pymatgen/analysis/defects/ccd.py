"""Configuration-coordinate diagram analysis."""
from __future__ import annotations

import logging
from ctypes import Structure
from dataclasses import dataclass
from itertools import groupby
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import numpy.typing as npt
from monty.json import MSONable
from pymatgen.electronic_structure.core import Spin
from pymatgen.io.vasp.outputs import WSWQ, BandStructure, Procar, Vasprun
from scipy.optimize import curve_fit

from .constants import AMU2KG, ANGS2M, EV2J, HBAR_EV, KB
from .recombination import get_SRH_coef
from .utils import get_localized_states, get_zfile, sort_positive_definite

# Copyright (c) Pymatgen Development Team.
# Distributed under the terms of the MIT License.

__author__ = "Jimmy Shen"
__copyright__ = "The Materials Project"
__maintainer__ = "Jimmy Shen"
__email__ = "jmmshn@gmail.com"
__date__ = "Mar 15, 2022"

_logger = logging.getLogger(__name__)


@dataclass
class HarmonicDefect(MSONable):
    """A class representing the a harmonic defect vibronic state.

    The vibronic part of a defect is often captured by a simple harmonic oscillator.
    This class store a representation of the SHO as well as some additional information for book-keeping purposes.

    Attributes:
        omega: The vibronic frequency of the phonon state in in the same units as the energy vs. Q plot.
        charge_state: The charge of the defect.
        ispin: The number of spin channels in the calculation (ISPIN from VASP).
        distortions: The distortion of the structure in units of amu^{-1/2} Angstrom^{-1}.
            This object's internal reference for the distortion should always be relaxed structure.
        structures: The list of structures that were used to compute the distortions.
        energies: The potential energy surface obtained by distorting the structure.A
        defect_band: The the index of the defect band since the defect for different
            kpoints and spins presented as `[(band, kpt, spin), ...]`.
        relaxed_indices: The indices of the relaxed defect structure.
        relaxed_bandstructure: The band structure of the relaxed defect calculation.
    """

    omega: float
    charge_state: int
    ispin: int
    distortions: Optional[list[float]] = None
    structures: Optional[list[Structure]] = None
    energies: Optional[list[float]] = None
    defect_band: Optional[Sequence[tuple]] = None
    relaxed_index: Optional[int] = None
    relaxed_bandstructure: Optional[BandStructure] = None

    def __repr__(self) -> str:
        """String representation of the harmonic defect."""
        return (
            f"HarmonicDefect("
            f"omega={self.omega:.3f} eV, "
            f"charge={self.charge_state}, "
            f"relaxed_index={self.relaxed_index}, "
            f"spin={self.spin_index}, "
            f"defect_band={self.defect_band}"
            ")"
        )

    @property
    def defect_band_index(self) -> int:
        """The index of the defect band."""
        bands = {band for band, _, _ in self.defect_band}
        if len(bands) != 1:
            raise ValueError("Defect band index is not unique.")
        return bands.pop()

    @property
    def spin_index(self) -> int:
        """The spin index of the defect.

        The integer spin index the defect state belongs to.
        0 for spin up and 1 for spin down. If ISPIN=1, this is always 0.
        """
        spins = {spin for _, _, spin in self.defect_band}
        if len(spins) != 1:
            raise ValueError("Spin index is not unique.")
        return spins.pop()

    @property
    def spin(self) -> Spin:
        """The spin of the defect returned as an Spin Enum."""
        if self.spin_index == 0:
            return Spin.up
        elif self.spin_index == 1:
            return Spin.down
        else:
            raise ValueError(f"Invalid spin index: {self.spin_index}")

    @property
    def relaxed_structure(self) -> Structure:
        """The relaxed structure."""
        return self.structures[self.relaxed_index]

    @classmethod
    def from_vaspruns(
        cls,
        vaspruns: list[Vasprun],
        charge_state: int,
        spin_index: int | None = None,
        relaxed_index: int | None = None,
        defect_band: Sequence[tuple] | None = None,
        procar: Procar | None = None,
        store_bandstructure: bool = False,
        get_band_structure_kwargs: dict | None = None,
        band_window: int = 7,
        **kwargs,
    ) -> HarmonicDefect:
        """Create a HarmonicDefect from a list of vasprun.

        .. note::
            The constructor check that you have the vaspruns sorted by the distortions
            but does not perform the sorting for you.  This serves as a safety check to
            ensure that the provided vaspruns are properly ordered.  This check might
            become optional in the future.

        Args:
            vaspruns: A list of Vasprun objects.
            charge_state: The charge state for the defect.
            relaxed_index: The index of the relaxed structure in the list of structures.
            defect_band: The the index of the defect band since the defect for different
                kpoints and spins presented as `[(band, kpt, spin), ...]`.
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

        energy_struct = list(map(_parse_vasprun, vaspruns))
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

        get_band_structure_kwargs = get_band_structure_kwargs or {}
        bandstructure = vaspruns[relaxed_index].get_band_structure(
            **get_band_structure_kwargs
        )
        ispin = vaspruns[relaxed_index].parameters["ISPIN"]

        if store_bandstructure:
            bs = bandstructure
        else:
            bs = None

        if defect_band is None:
            if procar is None:  # pragma: no cover
                raise ValueError(
                    "If defect_band_index is not provided, you must provide a Procar object."
                )
            # Get the defect bands
            defect_band_2s = list(
                get_localized_states(
                    bandstructure=bandstructure, procar=procar, band_window=band_window
                )
            )
            defect_band_2s.sort(key=lambda x: (x[2], x[1]))
            # group by the spin index
            defect_band_grouped = {
                spin: list(bands)
                for spin, bands in groupby(defect_band_2s, lambda x: x[2])
            }
            avg_localization = {
                spin: np.average([val for _, _, _, val in bands])
                for spin, bands in defect_band_grouped.items()
            }
            if spin_index is None:
                # get the most localized spin
                spin_index = min(avg_localization, key=avg_localization.get)
            # drop the val
            defect_band = [r_[:3] for r_ in defect_band_grouped[spin_index]]

        return cls(
            omega=omega,
            charge_state=charge_state,
            ispin=ispin,
            structures=structures,
            distortions=distortions,
            energies=energies,
            defect_band=defect_band,
            relaxed_index=relaxed_index,
            relaxed_bandstructure=bs,
            **kwargs,
        )

    @classmethod
    def from_directories(
        cls,
        directories: list[Path],
        charge_state: int | None = None,
        relaxed_index: int | None = None,
        defect_band: Sequence[tuple] | None = None,
        store_bandstructure: bool = False,
        get_band_structure_kwargs: dict | None = None,
        **kwargs,
    ) -> HarmonicDefect:
        """Create a HarmonicDefect from a list of directories.

        Args:
            directories: A list of directories.
            charge_state: The charge state for the defect. If None, the charge state
                will be determined using the data in the vasprun.xml and POTCAR files.
            relaxed_index: The index of the relaxed structure in the list of structures.
            defect_band: The the index of the defect band since the defect for different
                kpoints and spins presented as `[(band, kpt, spin), ...]`.
            store_bandstructure: Whether to store the bandstructure of the relaxed defect calculation.
                Defaults to False to save space.
            get_band_structure_kwargs: Keyword arguments to pass to the ``get_band_structure`` method.
            **kwargs: Additional keyword arguments to pass to the constructor.

        Returns:
            A HarmonicDefect object.
        """
        vaspruns = [Vasprun(d / "vasprun.xml") for d in directories]
        min_idx = np.argmin([v.final_energy for v in vaspruns])
        min_dir = directories[min_idx]
        procar_file = get_zfile(min_dir, "PROCAR")
        procar = Procar(procar_file) if procar_file else None

        if charge_state is None:
            if vaspruns[0].final_structure._charge is None:
                raise ValueError(
                    "Charge state is not provided and cannot be parsed from the POTCAR."
                )
            charge_state = vaspruns[0].final_structure.charge

        if any(v.final_structure.charge != charge_state for v in vaspruns):
            raise ValueError("All vaspruns must have the same charge state.")

        return cls.from_vaspruns(
            vaspruns=vaspruns,
            charge_state=charge_state,
            relaxed_index=relaxed_index,
            defect_band=defect_band,
            procar=procar,
            store_bandstructure=store_bandstructure,
            get_band_structure_kwargs=get_band_structure_kwargs,
            **kwargs,
        )

    @property
    def omega_eV(self) -> float:
        """Get the vibronic frequency of the phonon state in (eV)."""
        return self.omega * HBAR_EV * np.sqrt(EV2J / (ANGS2M**2 * AMU2KG))

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
            npt.NDArray: The electron phonon matrix elements from the defect band to all other bands.
                The indices are: [spin, kpoint, band_i, band_j]
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


# @dataclass
# class RadiativeCatpture(MSONable):
#     """Representation of a radiative capture event.

#     Attributes:
#         initial_state: The initial state of the radiative capture event.
#         final_state: The final state of the radiative capture event.
#         dQ: The configuration coordinate change between the relaxed initial state and the final state.
#         waveder: The data from the WAVEDER file obtained with ``LOPTICS=.True.``.

#     """

#     initial_state: HarmonicDefect
#     final_state: HarmonicDefect
#     dQ: float
#     waveder: Waveder

#     def get_coeff(
#         self,
#         T: float | npt.ArrayLike,
#         dE: float,
#         omega_photon: float,
#         volume: float | None = None,
#         g: int = 1,
#         occ_tol: float = 1e-3,
#         n_band_edge: int = 1,
#     ):
#         """Calculate the SRH recombination coefficient."""
#         if volume is None:
#             volume = self.initial_state.relaxed_structure.volume

#         me_all = self.get_dipoles()  # indices: [band, kpoint, spin, coord]

#         istate = self.initial_state

#         if self.initial_state.charge_state == self.final_state.charge_state + 1:
#             band_slice = slice(
#                 istate.defect_band_index + 1, istate.defect_band_index + 1 + n_band_edge
#             )
#         elif self.initial_state.charge_state == self.final_state.charge_state - 1:
#             band_slice = slice(
#                 istate.defect_band_index - n_band_edge, istate.defect_band_index
#             )
#         else:
#             raise ValueError(
#                 "SRH capture event must involve a charge state change of 1."
#             )

#         me_band_edge = me_all[band_slice, istate.kpt_index, istate.spin_index]

#         return get_Rad_coef(
#             T,
#             dQ=self.dQ,
#             dE=dE,
#             omega_i=self.initial_state.omega_eV,
#             omega_f=self.final_state.omega_eV,
#             omega_photon=omega_photon,
#             dipole_me=np.average(me_band_edge),
#             volume=volume,
#             g=g,
#             occ_tol=occ_tol,
#         )

#     @classmethod
#     def from_directories(
#         cls,
#         initial_dirs: list[Path],
#         final_dirs: list[Path],
#         waveder_dir: Path,
#         kpt_index: int,
#         initial_charge_state: int | None = None,
#         final_charge_state: int | None = None,
#         spin_index: int | None = None,
#         defect_band_index: int | None = None,
#         store_bandstructure: bool = False,
#         get_band_structure_kwargs: dict | None = None,
#         **kwargs,
#     ) -> RadiativeCatpture:
#         """Create a RadiativeCapture object from a list of directories.

#         Args:
#             initial_dirs: A list of directories for the initial state.
#             final_dirs: A list of directories for the final state.
#             waveder_dir: The directory containing the WAVEDER file.
#             kpt_index: The index of the k-point to use.
#             initial_charge_state: The charge state of the initial state.
#                 If None, the charge state is determined from the vasprun.xml and POTCAR files.
#             final_charge_state: The charge state of the final state.
#                 If None, the charge state is determined from the vasprun.xml and POTCAR files.
#             spin_index: The index of the spin channel to use.
#                 If None, the spin channel is determined by the channel with the most localized state.
#             defect_band_index: The index of the defect band (0-indexed).
#                 If None, the defect band is determined by the band with the most localized state.
#             store_bandstructure: Whether to store the band structure.
#             get_band_structure_kwargs: Keyword arguments to pass to get_band_structure.
#             **kwargs: Keyword arguments to pass to the HarmonicDefect constructor.

#         Returns:
#             A SRHCapture object.
#         """
#         initial_defect = HarmonicDefect.from_directories(
#             directories=initial_dirs,
#             charge_state=initial_charge_state,
#             spin_index=spin_index,
#             relaxed_index=None,
#             defect_band_index=defect_band_index,
#             store_bandstructure=store_bandstructure,
#             get_band_structure_kwargs=get_band_structure_kwargs,
#             **kwargs,
#         )

#         # the final state does not need the additional
#         # information about the electronic states
#         final_defect = HarmonicDefect.from_directories(
#             directories=final_dirs,
#             kpt_index=kpt_index,
#             charge_state=final_charge_state,
#             spin_index=spin_index,
#             relaxed_index=None,
#             defect_band_index=None,
#             store_bandstructure=None,
#             get_band_structure_kwargs=None,
#             **kwargs,
#         )

#         waveder_file = get_zfile(waveder_dir, "WAVEDER")
#         waveder = Waveder(waveder_file)
#         dQ = get_dQ(initial_defect.relaxed_structure, final_defect.relaxed_structure)
#         return cls(initial_defect, final_defect, dQ=dQ, waveder=waveder)

#     def get_dipoles(self) -> npt.NDArray:
#         """Get the dipole matrix elements associated with the defect.

#         Return

#         Returns:
#             The dipole matrix elements for the defect. The indices are:
#                 ``[band, k-point, spin, cart. direction]``.
#         """
#         return self.waveder.cder_data[self.initial_state.defect_band_index, ...]

#     def get_spectra(self) -> npt.NDArray:
#         """Get the spectra for the defect.

#         Compute the energy differences between all the bands and he defect band.

#         Returns:
#             Array of size ``[n_bands, n_kpoints, n_spins]``.
#         """
#         return self.initial_state._get_ediff(output_order="bks")


@dataclass
class SRHCapture(MSONable):
    """Representation of a SRH capture event.

    Performs book keeping of initial and final states.

    Attributes:
        initial_state: The initial state of the SRH capture event.
        final_state: The final state of the SRH capture event.
        dQ: The distortion between the structures in units of amu^{-1/2} Angstrom^{-1}.
            By convention, the final state should be on the +dQ side of the initial state.
            This should only matter once we start considering anharmonic defects.
        wswqs: The PAW overlap matrix element <W(0)|S|W(Q)> stored as a list of WSWQ objects.
    """

    initial_state: HarmonicDefect
    final_state: HarmonicDefect
    dQ: float
    wswqs: list[WSWQ]

    def get_coeff(
        self,
        T: float | npt.ArrayLike,
        dE: float,
        kpt_index: int,
        volume: float | None = None,
        g: int = 1,
        occ_tol: float = 1e-3,
        n_band_edge: int = 1,
    ) -> npt.NDArray:
        """Calculate the SRH recombination coefficient.

        Args:
            T: The temperature in Kelvin.
            dE: The energy difference between the defect and the conduction band.
            kpt_index: The index of the k-point to use, this should be
                determined by the band edge so most likely this will be Gamma.
            volume: The volume of the structure in Angstrom^3.
                If None, the volume of the initial state is used.
            g: The degeneracy of the defect state.
            occ_tol: The tolerance for determining if a state is occupied.
            n_band_edge: The number of bands to average over at the band edge.

        Returns:
            The SRH recombination coefficient in units of cm^3 s^-1.
        """
        if volume is None:
            volume = self.initial_state.relaxed_structure.volume
        elph_me_all = self.initial_state.get_elph_me(
            self.wswqs
        )  # indices: [spin, kpoint, band_i, band_j]
        istate = self.initial_state

        if self.initial_state.charge_state == self.final_state.charge_state + 1:
            band_slice = slice(
                istate.defect_band_index + 1, istate.defect_band_index + 1 + n_band_edge
            )
        elif self.initial_state.charge_state == self.final_state.charge_state - 1:
            band_slice = slice(
                istate.defect_band_index - n_band_edge, istate.defect_band_index
            )
        else:
            raise ValueError(
                "SRH capture event must involve a charge state change of 1."
            )

        elph_me_band_edge = elph_me_all[istate.spin_index, kpt_index, band_slice]

        return get_SRH_coef(
            T,
            dQ=self.dQ,
            dE=dE,
            omega_i=self.initial_state.omega_eV,
            omega_f=self.final_state.omega_eV,
            elph_me=np.average(elph_me_band_edge),
            volume=volume,
            g=g,
            occ_tol=occ_tol,
        )

    @classmethod
    def from_directories(
        cls,
        initial_dirs: list[Path],
        final_dirs: list[Path],
        wswq_dir: Path,
        initial_charge_state: int | None = None,
        final_charge_state: int | None = None,
        defect_band: list[tuple] | None = None,
        store_bandstructure: bool = False,
        get_band_structure_kwargs: dict | None = None,
        **kwargs,
    ) -> SRHCapture:
        """Create a SRHCapture from a list of directories.

        Args:
            initial_dirs: A list of directories for the initial state.
            final_dirs: A list of directories for the final state.
            wswq_dir: The directory containing the WSWQ files.
            initial_charge_state: The charge state of the initial state.
                If None, the charge state is determined from the vasprun.xml and POTCAR files.
            final_charge_state: The charge state of the final state.
                If None, the charge state is determined from the vasprun.xml and POTCAR files.
            defect_band: The the index of the defect band since the defect for different
                kpoints and spins presented as `[(band, kpt, spin), ...]`.
            store_bandstructure: Whether to store the band structure.
            get_band_structure_kwargs: Keyword arguments to pass to get_band_structure.
            **kwargs: Keyword arguments to pass to the HarmonicDefect constructor.

        Returns:
            A SRHCapture object.
        """
        initial_defect = HarmonicDefect.from_directories(
            directories=initial_dirs,
            charge_state=initial_charge_state,
            relaxed_index=None,
            defect_band=defect_band,
            store_bandstructure=store_bandstructure,
            get_band_structure_kwargs=get_band_structure_kwargs,
            **kwargs,
        )

        # the final state does not need the additional
        # information about the electronic states
        final_defect = HarmonicDefect.from_directories(
            directories=final_dirs,
            charge_state=final_charge_state,
            relaxed_index=None,
            defect_band=((),),  # Skip the procar check since we only need the energies.
            store_bandstructure=None,
            get_band_structure_kwargs=None,
            **kwargs,
        )
        wswq_files = [f for f in wswq_dir.glob("WSWQ*")]
        wswq_files.sort(
            key=lambda x: int(x.stem.split(".")[1])
        )  # does stem work for non-zipped files?
        wswqs = [WSWQ.from_file(f) for f in wswq_files]
        dQ = get_dQ(initial_defect.relaxed_structure, final_defect.relaxed_structure)
        return cls(initial_defect, final_defect, dQ=dQ, wswqs=wswqs)


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

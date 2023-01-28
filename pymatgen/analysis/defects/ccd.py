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
from matplotlib.axes import Axes
from monty.json import MSONable
from pymatgen.electronic_structure.core import Spin
from pymatgen.io.vasp.optics import DielectricFunctionCalculator
from pymatgen.io.vasp.outputs import WSWQ, BandStructure, Procar, Vasprun, Waveder
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
        vrun: The Vasprun object for the relaxed defect structure.
        distortions: The distortion of the structure in units of amu^{-1/2} Angstrom^{-1}.
            This object's internal reference for the distortion should always be relaxed structure.
        structures: The list of structures that were used to compute the distortions.
        energies: The potential energy surface obtained by distorting the structure.A
        defect_band: The the index of the defect band since the defect for different
            kpoints and spins presented as `[(band, kpt, spin), ...]`.
        relaxed_indices: The indices of the relaxed defect structure.
        relaxed_bandstructure: The band structure of the relaxed defect calculation.
        wswqs: Dict of WSWQ objects for each distortion. The key is the distortion.
        waveder: The Waveder object for the relaxed defect structure.
    """

    omega: float
    charge_state: int
    ispin: int
    vrun: Optional[Vasprun] = None
    distortions: Optional[list[float]] = None
    structures: Optional[list[Structure]] = None
    energies: Optional[list[float]] = None
    defect_band: Optional[Sequence[tuple]] = None
    relaxed_index: Optional[int] = None
    relaxed_bandstructure: Optional[BandStructure] = None
    wswqs: Optional[list[dict]] = None
    waveder: Optional[Waveder] = None

    def __repr__(self) -> str:
        """String representation of the harmonic defect."""
        return (
            f"HarmonicDefect("
            f"omega={self.omega_eV:.3f} eV, "
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
            vrun=vaspruns[relaxed_index],
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
            directories: A list of directories containing the vasprun.xml files.
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

    def read_wswqs(
        self, directory: Path, distortions: list[float] | None = None
    ) -> None:
        """Read the WSWQ files from a directory.

        Assuming that we have a directory containing the WSWQ files ordered in
        the same way as the `self.distortions`.

        Args:
            directory: The directory containing the WSWQ files formatted as WSWQ.0, WSWQ.1, ...
            distortions: The distortions used to generate the WSWQ files,
                if different from self.distortions
        """
        wswq_files = [f for f in directory.glob("WSWQ*")]
        wswq_files.sort(key=lambda x: int(x.name.split(".")[1]))
        if distortions is None:
            distortions = self.distortions

        if len(wswq_files) != len(distortions):
            raise ValueError(
                f"Number of WSWQ files ({len(wswq_files)}) does not match number of distortions ({len(distortions)})."
            )
        self.wswqs = [
            {"Q": d, "wswq": WSWQ.from_file(f)} for d, f in zip(distortions, wswq_files)
        ]

    def get_elph_me(self, defect_state: tuple) -> npt.ArrayLike:
        """Calculate the electron phonon matrix elements.

        Combine the data from the WSWQs to calculate the electron phonon matrix elements.
        The matrix elements are calculated by combining the finite difference from the matrix overlaps.

        (e_i - e_f) d(<W|S|W(Q)>) / dQ

        And the eigenvalue difference.

        Args:
            defect_state: The defect state as a tuple of (band, kpoint, spin).
                Note that even though the defect state is localized and does not have a well defined kpoint,
                the band-edge states are usually dispersive and the k-point provided here should match the
                k-point associated with the band-edge states.

        Returns:
            npt.ArrayLike: The electron phonon matrix elements from the defect band to all other bands.
                The indices are [band_j,]
        """
        if self.wswqs is None:
            raise RuntimeError("WSWQs have not been read. Use `read_wswqs` first.")
        distortions = [wswq["Q"] for wswq in self.wswqs]
        wswqs = [wswq["wswq"] for wswq in self.wswqs]
        band_index, kpoint_index, spin_index = defect_state
        # The second band index is associated with the defect state
        # since we are usually interested in capture
        slopes = _get_wswq_slope(distortions, wswqs)[
            spin_index, kpoint_index, :, band_index
        ]
        ediffs = self._get_ediff(output_order="skb")[spin_index, kpoint_index, :]
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
        if self.relaxed_bandstructure is None:
            raise ValueError(  # pragma: no cover
                "The ``relaxed_bandstructure`` must be set before ``ediff`` can be computed. "
                "Try setting ``store_bandstructure=True`` when initializing."
            )

        ediffs_ = _get_ks_ediff(
            bandstructure=self.relaxed_bandstructure,
            defect_band=self.defect_band,
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

    def get_dielectric_function(
        self, idir: int, jdir: int
    ) -> tuple[npt.ArrayLike, npt.ArrayLike, npt.ArrayLike]:
        """Calculate the dielectric function.

        Args:
            idir: The first direction of the dielectric tensor.
            jdir: The second direction of the dielectric tensor.

        Returns:
            energy: The energy grid representing the dielectric function.
            eps_vbm: The dielectric function from the VBM to the defect state.
            eps_cbm: The dielectric function from the defect state to the CBM.
        """
        dfc = DielectricFunctionCalculator.from_vasp_objects(
            vrun=self.vrun, waveder=self.waveder
        )

        # two masks to select for VBM -> Defect and Defect -> CBM
        mask_vbm = np.zeros_like(dfc.cder_real)
        mask_cbm = np.zeros_like(dfc.cder_real)
        min_def_eig, max_def_eig = np.inf, -np.inf
        for ib, ik, ispin in self.defect_band:
            mask_vbm[:ib, ib, ik, ispin] = 1.0
            mask_cbm[ib, ib:, ik, ispin] = 1.0
            min_def_eig = min(dfc.eigs[ib, ik, ispin], min_def_eig)
            max_def_eig = max(dfc.eigs[ib, ik, ispin], max_def_eig)

        # VBM must be lower than defect and CBM must be higher than defect
        # For situations where there are multiple defect states, hopefully
        # the Fermi smearing will reduce the transition between defect states
        # and we will only measure transitions to the band edges.
        e_vbm = np.max(dfc.eigs[dfc.eigs < min_def_eig])
        e_cbm = np.min(dfc.eigs[dfc.eigs > max_def_eig])

        fermi_vbm = (e_vbm + min_def_eig) / 2
        fermi_cbm = (e_cbm + max_def_eig) / 2

        energy, eps_vbm = dfc.get_epsilon(idir, jdir, fermi_vbm, mask=mask_vbm)
        _, eps_cbm = dfc.get_epsilon(idir, jdir, fermi_cbm, mask=mask_cbm)

        return energy, eps_vbm, eps_cbm

    # def get_dipoles(self, defect_state: tuple[int, int, int]) -> npt.NDArray:
    #     """Get the dipole matrix elements associated with the defect.

    #     Return

    #     Returns:
    #         The dipole matrix elements for the defect. The indices are:
    #             ``[band, k-point, spin, cart. direction]``.
    #     """
    #     defect_band, defect_kpt, defect_spin = defect_state
    #     all_me = self.waveder.cder_data[:, defect_kpt, defect_spin, :]

    # def get_spectra(self) -> npt.NDArray:
    #     """Get the spectra for the defect.

    #     Compute the energy differences between all the bands and he defect band.

    #     Returns:
    #         Array of size ``[n_bands, n_kpoints, n_spins]``.
    #     """
    #     return self.initial_state._get_ediff(output_order="bks")


def get_SRH_coefficient(
    initial_state: HarmonicDefect,
    final_state: HarmonicDefect,
    defect_state: tuple[int, int, int],
    T: float | npt.ArrayLike,
    dE: float,
    g: int = 1,
    occ_tol: float = 1e-3,
    n_band_edge: int = 1,
    use_final_state_elph: bool = False,
) -> npt.ArrayLike:
    """Get the SRH coefficient for a defect.

    Args:
        initial_state: The initial charge state of the defect.
        final_state: The final charge state of the defect.
        defect_state: The band, k-point, and spin of the defect.
        T: The temperature in Kelvin.
        dE: The energy difference between the defect and the band edge.
        g: The degeneracy of the defect state.
        occ_tol: The tolerance for determining if a state is occupied.
        n_band_edge: The number of bands to average over at the band edge.
        use_final_state_elph: Whether to use the final state's ELPH data.
            This is useful if the initial state does not have a well-defined
            defect state.

    Returns:
        The SRH recombination coefficient in units of cm^3 s^-1.
    """
    if use_final_state_elph:
        me_all = final_state.get_elph_me(defect_state=defect_state)
    else:
        me_all = initial_state.get_elph_me(defect_state=defect_state)
    defect_band, _, _ = defect_state

    if initial_state.charge_state == final_state.charge_state + 1:
        band_slice = slice(defect_band + 1, defect_band + 1 + n_band_edge)
    elif initial_state.charge_state == final_state.charge_state - 1:
        band_slice = slice(defect_band - n_band_edge, defect_band)
    else:
        raise ValueError("SRH capture event must involve a charge state change of 1.")

    me_band_edge = me_all[band_slice]
    dQ = get_dQ(initial_state.relaxed_structure, final_state.relaxed_structure)
    volume = initial_state.relaxed_structure.volume
    return get_SRH_coef(
        T,
        dQ=dQ,
        dE=dE,
        omega_i=initial_state.omega_eV,
        omega_f=final_state.omega_eV,
        elph_me=np.average(me_band_edge),
        volume=volume,
        g=g,
        occ_tol=occ_tol,
    )


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
            Since there is always ambiguity in the phase, we require that the output
            is always positive.
    """
    yy = np.stack([np.abs(ww.data) * np.sign(qq) for qq, ww in zip(distortions, wswqs)])
    _, *oldshape = yy.shape
    return np.polyfit(distortions, yy.reshape(yy.shape[0], -1), deg=1)[0].reshape(
        *oldshape
    )


def _get_ks_ediff(
    bandstructure: BandStructure,
    defect_band: Sequence[tuple],
) -> dict[Spin, npt.NDArray]:
    """Calculate the Kohn-Sham energy between the defect state.

    Get the eigenvalue differences to the defect band. Report this difference
    on each k-point and each spin, the result should be shape [nspins, nkpoints, nbands].

    Args:
        bandstructure: A BandStructure object.
        defect_band: The defect band given as a list of tuples (band_index, kpoint_index, spin_index).

    Returns:
        npt.NDArray: The Kohn-Sham energy difference between the defect state and other states.
        Indexed the same way as ``bandstructure.bands``.
    """
    res = dict()
    b_at_kpt_and_spin = {(k, s): b for b, k, s in defect_band}
    for ispin, eigs in bandstructure.bands.items():
        spin_index = 0 if ispin == Spin.up else 1
        res[ispin] = np.zeros_like(eigs)
        for ikpt, kpt in enumerate(bandstructure.kpoints):
            iband = b_at_kpt_and_spin.get((ikpt, spin_index), None)
            # import ipdb; ipdb.set_trace()
            if iband is None:
                continue
            e_at_def_band = eigs[iband, ikpt]
            e_diff = eigs[:, ikpt] - e_at_def_band
            res[ispin][:, ikpt] = e_diff
    return res


def plot_pes(
    hd: HarmonicDefect, x_shift=0, y_shift=0, width: float = 1.0, ax: Axes = None
) -> None:
    """Plot the Potential Energy Surface of a HarmonicDefect.

    Args:
        hd: HarmonicDefect object
        x_shift: shift the PES by this amount in the x-direction
        y_shift: shift the PES by this amount in the y-direction

    Returns:
        None
    """
    if ax is None:
        from matplotlib import pyplot as plt

        fig, ax = plt.subplots()
    ax.plot(
        np.array(hd.distortions) + x_shift,
        (np.array(hd.energies) - hd.energies[hd.relaxed_index]) + y_shift,
        "o",
        ms=10,
    )
    xx = np.linspace(-width / 2, width / 2, 20)
    yy = 0.5 * hd.omega**2 * xx**2
    ax.plot(xx + x_shift, (yy + y_shift))
    ax.set_xlabel("Q [amu$^{1/2}$Å]")
    ax.set_ylabel("Energy [eV]")

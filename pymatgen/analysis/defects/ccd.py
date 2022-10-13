"""Configuration-coordinate diagram analysis."""
from __future__ import annotations

import logging
from ctypes import Structure
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import numpy.typing as npt
from monty.json import MSONable
from pymatgen.electronic_structure.core import Spin
from pymatgen.io.vasp.outputs import WSWQ, BandStructure, Procar, Vasprun
from scipy.optimize import curve_fit

from pymatgen.analysis.defects.recombination import get_SRH_coef

from .constants import AMU2KG, ANGS2M, EDEPS, EV2J, HBAR_EV, KB
from .utils import get_localized_state, get_zfile, sort_positive_definite

# Copyright (c) Pymatgen Development Team.
# Distributed under the terms of the MIT License.

__author__ = "Jimmy Shen"
__copyright__ = "The Materials Project"
__maintainer__ = "Jimmy Shen"
__email__ = "jmmshn@gmail.com"
__date__ = "Mar 15, 2022"

_logger = logging.getLogger(__name__)


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
        kpt_index: The kpoint index in the simulation that correspond to the band edge.
        spin_index: The spin index in the simulation that correspond to the band edge.
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
    kpt_index: int
    spin_index: int
    distortions: Optional[list[float]] = None
    structures: Optional[list[Structure]] = None
    energies: Optional[list[float]] = None
    defect_band_index: Optional[int] = None
    relaxed_index: Optional[int] = None
    relaxed_bandstructure: Optional[BandStructure] = None

    def __repr__(self) -> str:
        """String representation of the harmonic defect."""
        return (
            f"HarmonicDefect("
            f"omega={self.omega:.3f} eV, "
            f"charge={self.charge_state}, "
            f"relaxed_index={self.relaxed_index}, "
            f"kpt={self.kpt_index}, "
            f"spin={self.spin_index}, "
            f"defect_band_index={self.defect_band_index}"
            ")"
        )

    @property
    def spin(self) -> Spin:
        """The spin of the defect."""
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
        kpt_index: int,
        charge_state: int,
        spin_index: int | None = None,
        relaxed_index: int | None = None,
        defect_band_index: int | None = None,
        procar: Procar | None = None,
        store_bandstructure: bool = False,
        get_band_structure_kwargs: dict | None = None,
        **kwargs,
    ) -> HarmonicDefect:
        """Create a HarmonicDefect from a list of vasprun.

        .. note::
            The constructor check that you have the vaspruns sorted by the distortions
            but does not perform the sorting for you.  This serves as a safety check to
            ensure that the vaspruns are properly ordered.

        Args:
            vaspruns: A list of Vasprun objects.
            charge_state: The charge state for the defect.
            kpt_index: The index of the kpoint that corresponds to the band edge.
            spin_index: The index of the spin that corresponds to the band edge.
                If None, we will assume that the band edge is spin-independent and we will use the
                spin channel with the most localized state.
            relaxed_index: The index of the relaxed structure in the list of structures.
            defect_band_index: The index of the defect band (0-indexed).  This is found by looking
                at the inverse participation ratio of the different states.
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
        if store_bandstructure:
            bs = bandstructure
        else:
            bs = None

        if defect_band_index is None:
            if procar is None:  # pragma: no cover
                raise ValueError(
                    "If defect_band_index is not provided, you must provide a Procar object."
                )
            loc_res = get_localized_state(
                bandstructure=bandstructure, procar=procar, k_index=kpt_index
            )
            spin_e, (defect_band_index, _) = min(loc_res.items(), key=lambda x: x[1])
        else:
            if spin_index is None:
                raise ValueError(
                    "If ``defect_band_index`` is provided, you must provide also ``spin_index``."
                )
            if kpt_index is None:  # pragma: no cover
                raise ValueError(
                    "If ``defect_band_index`` is provided, you must provide also ``kpt_index``."
                )

        if spin_index is None:
            spin_index = 0 if spin_e == Spin.up else 1

        return cls(
            omega=omega,
            charge_state=charge_state,
            kpt_index=kpt_index,
            spin_index=spin_index,
            structures=structures,
            distortions=distortions,
            energies=energies,
            defect_band_index=defect_band_index,
            relaxed_index=relaxed_index,
            relaxed_bandstructure=bs,
            **kwargs,
        )

    @classmethod
    def from_directories(
        cls,
        directories: list[Path],
        kpt_index: int,
        charge_state: int | None = None,
        spin_index: int | None = None,
        relaxed_index: int | None = None,
        defect_band_index: int | None = None,
        store_bandstructure: bool = False,
        get_band_structure_kwargs: dict | None = None,
        **kwargs,
    ) -> HarmonicDefect:
        """Create a HarmonicDefect from a list of directories.

        Args:
            directories: A list of directories.
            kpt_index: The index of the kpoint that corresponds to the band edge.
            charge_state: The charge state for the defect. If None, we will try to parse the POTCAR.
            spin_index: The index of the spin that corresponds to the band edge.
                If None, we will assume that the band edge is spin-independent and we will use the
                spin channel with the most localized state.
            relaxed_index: The index of the relaxed structure in the list of structures.
            defect_band_index: The index of the defect band (0-indexed).  This is found by looking
                at the inverse participation ratio of the different states.
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
            kpt_index=kpt_index,
            charge_state=charge_state,
            spin_index=spin_index,
            relaxed_index=relaxed_index,
            defect_band_index=defect_band_index,
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
# class OpticalHarmonicDefect(HarmonicDefect):
#     """Representation of Harmonic defect with optical (dipole) matrix elements.

#     The dipole matrix elements are computed by VASP and reported in the WAVEDER file.

#     Attributes:
#         omega: The vibronic frequency of the phonon state in in the same units as the energy vs. Q plot.
#         charge: The charge state. This should be the charge of the defect
#             simulation that gave rise to the minimum of the parabola.
#         distortions: The distortion of the structure in units of amu^{-1/2} Angstrom^{-1}.
#             This object's internal reference for the distoration should always be relaxed structure.
#         structures: The list of structures that were used to compute the distortions.
#         energies: The potential energy surface obtained by distorting the structure.A
#         defect_band_index: The index of the defect band.
#         relaxed_indices: The indices of the relaxed defect structure.
#         relaxed_bandstructure: The band structure of the relaxed defect calculation.
#         waveder: The WAVEDER object containing the dipole matrix elements.
#     """

#     # TODO: use kw_only once we drop Python < 3.10
#     waveder: Waveder | None = None

#     @classmethod
#     def from_vaspruns_and_waveder(
#         cls,
#         vaspruns: list[Vasprun],
#         waveder: Waveder,
#         charge_state: int,
#         relaxed_index: int | None = None,
#         defect_band_index: int | None = None,
#         procar: Procar | None = None,
#         get_band_structure_kwargs: dict | None = None,
#         **kwargs,
#     ) -> OpticalHarmonicDefect:
#         """Create a HarmonicDefectPhonon from a list of vasprun.

#         .. note::
#             The constructor check that you have the vaspruns sorted by the distortions
#             but does not order it for you.

#         Args:
#             vaspruns: A list of Vasprun objects.
#             charge_state: The charge state for the defect.
#             relaxed_index: The index of the relaxed structure in the list of structures.
#             defect_band_index: The index of the defect band (0-indexed).
#             procar: The Procar object for the defect calculation.

#         Returns:
#             An OpticalHarmonicDefect object.
#         """
#         obj = super().from_vaspruns(
#             vaspruns,
#             charge_state,
#             relaxed_index,
#             waveder=waveder,
#             defect_band_index=defect_band_index,
#             procar=procar,
#             store_bandstructure=True,
#             get_band_structure_kwargs=get_band_structure_kwargs,
#             **kwargs,
#         )
#         if obj.defect_band_index is None:
#             raise ValueError(  # pragma: no cover
#                 "You must provide `defect_band_index` or PROCAR to help indetify the `defect_band_index`."
#             )
#         if obj.relaxed_bandstructure is None:
#             raise ValueError(  # pragma: no cover
#                 "The bandstructure was not populated properly check the constructor of the parent."
#             )
#         return obj

#     @classmethod
#     def from_vaspruns(
#         cls,
#         *args,
#         **kwargs,
#     ) -> HarmonicDefect:  # noqa
#         """Not implemented."""
#         raise NotImplementedError("Use from_vaspruns_and_waveder instead.")

#     def _get_defect_dipoles(self) -> npt.NDArray:
#         """Get the dipole matrix elements for the defect.

#         Returns:
#             The dipole matrix elements for the defect. The indices are:
#                 ``[band index, k-point index, spin index, cart. direction]``.
#         """
#         return self.waveder.cder_data[self.defect_band_index, ...]

#     def _get_spectra(self) -> npt.NDArray:
#         """Get the spectra for the defect.

#         Args:
#             bandstructure: The band structure of the relaxed defect calculation.
#             shift: The shift to apply to the spectra.
#         """
#         return self._get_ediff(output_order="bks")


@dataclass
class SRHCapture(MSONable):
    """Representation of SRH capture event.

    Performs book keeping of initial and final states.

    Args:
        initial_state: The initial state of the SRH capture event.
        final_state: The final state of the SRH capture event.
        dQ: The distortion between the structures in units of amu^{-1/2} Angstrom^{-1}.
            By convention, the final state should be on the +dQ side of the initial state.
            This should only matter once we start considering anharmonic defects.
    """

    initial_state: HarmonicDefect
    final_state: HarmonicDefect
    dQ: float
    wswqs: list[WSWQ]

    def get_coeff(
        self,
        T: float | npt.ArrayLike,
        dE: float,
        volume: float | None = None,
        g: int = 1,
        occ_tol: float = 1e-3,
        n_band_edge: int = 1,
    ):
        """Calculate the SRH recombination coefficient."""
        if volume is None:
            volume = self.initial_state.relaxed_structure.volume
        elph_me_all = self.initial_state.get_elph_me(
            self.wswqs
        )  # indices: [spin, kpoint, band_i, band_j]
        istate = self.initial_state

        if self.initial_state.charge_state == self.final_state.charge_state + 1:
            sl_bands = slice(
                istate.defect_band_index + 1, istate.defect_band_index + 1 + n_band_edge
            )
        elif self.initial_state.charge_state == self.final_state.charge_state - 1:
            sl_bands = slice(
                istate.defect_band_index - n_band_edge, istate.defect_band_index
            )
        else:
            raise ValueError(
                "SRH capture event must involve a charge state change of 1."
            )

        elph_me_band_edge = elph_me_all[istate.spin_index, istate.kpt_index, sl_bands]

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
        kpt_index: int,
        initial_charge_state: int | None = None,
        final_charge_state: int | None = None,
        spin_index: int | None = None,
        defect_band_index: int | None = None,
        store_bandstructure: bool = False,
        get_band_structure_kwargs: dict | None = None,
        **kwargs,
    ) -> SRHCapture:
        """Create a SRHCapture from a list of directories.

        Args:
            initial_dirs: A list of directories for the initial state.
            final_dirs: A list of directories for the final state.
            dQ: The distortion between the structures in units of amu^{-1/2} Angstrom^{-1}.
                By convention, the final state should be on the +dQ side of the initial state.
                This should only matter once we start considering anharmonic defects.
            charge_state: The charge state for the defect.
            relaxed_index: The index of the relaxed structure in the list of structures.
            defect_band_index: The index of the defect band (0-indexed).

        Returns:
            A SRHCapture object.
        """
        initial_defect = HarmonicDefect.from_directories(
            directories=initial_dirs,
            kpt_index=kpt_index,
            charge_state=initial_charge_state,
            spin_index=spin_index,
            relaxed_index=None,
            defect_band_index=defect_band_index,
            store_bandstructure=store_bandstructure,
            get_band_structure_kwargs=get_band_structure_kwargs,
            **kwargs,
        )

        # the final state does not need the additional
        # information about the electronic states
        final_defect = HarmonicDefect.from_directories(
            directories=final_dirs,
            kpt_index=kpt_index,
            charge_state=final_charge_state,
            spin_index=spin_index,
            relaxed_index=None,
            defect_band_index=None,
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

"""Defect corrections."""

from __future__ import annotations

import logging
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import scipy
from numpy.typing import ArrayLike, NDArray
from scipy import stats

from pymatgen.analysis.defect.thermo import DefectEntry
from pymatgen.analysis.defect.utils import (
    QModel,
    ang_to_bohr,
    converge,
    eV_to_k,
    generate_reciprocal_vectors_squared,
    hart_to_ev,
)

__author__ = "Jimmy-Xuan Shen, Danny Broberg, Shyam Dwaraknath"
__copyright__ = "Copyright 2022, The Materials Project"
__maintainer__ = "Jimmy-Xuan Shen"
__email__ = "jmmshn@gmail.com"

_logger = logging.getLogger(__name__)


class FreysoldtCorrection:
    """A class for FreysoldtCorrection class.

    Note: This class should NEVER be serialized since bulk locpot is shared between the different
    defect entries.

    Largely adapted from PyCDT code
    If this correction is used, please reference Freysoldt's original paper.
    doi: 10.1103/PhysRevLett.102.016402
    """

    def __init__(
        self,
        dielectric_const: float | NDArray,
        q_model: Optional[QModel] = None,
        energy_cutoff: float = 520,
        mad_tol: float = 0.0001,
    ):
        """Initializes the FreysoldtCorrection class.

        Args:
        dielectric_const: Dielectric constant of the material.
        q_model: A QModel object. ...
        energy_cutoff: Maximum energy in eV in reciprocal space to perform integration for potential correction.
        mad_tol:Convergence criteria for the Madelung energy for potential correction
        """
        self.q_model = QModel() if not q_model else q_model
        self.energy_cutoff = energy_cutoff
        self.mad_tol = mad_tol
        self.dielectric_const = dielectric_const

        if isinstance(dielectric_const, (int, float)):
            self.dielectric = float(dielectric_const)
        else:
            self.dielectric = float(np.mean(np.diag(dielectric_const)))

        self.metadata: dict = {"pot_plot_data": {}, "pot_corr_uncertainty_md": {}}

    def get_correction(self, defect_entry: DefectEntry, defect_frac_coords: ArrayLike):
        """Gets the Freysoldt correction for a defect entry.

        Args:
            defect_entry (DefectEntry): defect entry to compute Freysoldt correction on.
            defect_frac_coords (ArrayLike): fractional coordinates of the defect.

        Returns:
            FreysoldtCorrection values as a dictionary
        """
        list_axis_grid = np.array([*map(defect_entry.defect_locpot.get_axis_grid, [0, 1, 2])])
        list_defect_plnr_avg_esp = np.array([*map(defect_entry.defect_locpot.get_average_along_axis, [0, 1, 2])])
        list_bulk_plnr_avg_esp = np.array([*map(defect_entry.bulk_locpot.get_average_along_axis, [0, 1, 2])])
        list_axes = range(len(list_axis_grid))

        lattice = defect_entry.sc_entry.structure.lattice.copy()

        q = defect_entry.charge_state
        print(q)
        es_corr = self.perform_es_corr(lattice, q)

        pot_corr_tracker = []

        for x, pureavg, defavg, axis in zip(
            list_axis_grid, list_bulk_plnr_avg_esp, list_defect_plnr_avg_esp, list_axes
        ):
            tmp_pot_corr = self.perform_pot_corr(
                x,
                pureavg,
                defavg,
                lattice,
                q,
                defect_frac_coords,
                axis,
                widthsample=1.0,
            )
            pot_corr_tracker.append(tmp_pot_corr)

        pot_corr = np.mean(pot_corr_tracker)

        # defect_entry.parameters["freysoldt_meta"] = dict(self.metadata)
        # defect_entry.parameters["potalign"] = pot_corr / (-q) if q else 0.0

        return {
            "freysoldt_electrostatic": es_corr,
            "freysoldt_potential_alignment": pot_corr,
        }

    def perform_es_corr(self, lattice, q, step=1e-4):
        """Perform Electrostatic Freysoldt Correction.

        Args:
            lattice: Pymatgen lattice object
            q (int): Charge of defect
            step (float): step size for numerical integration

        Return:
            Electrostatic Point Charge contribution to Freysoldt Correction (float)
        """
        _logger.info("Running Freysoldt 2011 PC calculation (should be equivalent to sxdefectalign)")
        _logger.debug("defect lattice constants are (in angstroms)" + str(lattice.abc))

        [a1, a2, a3] = ang_to_bohr * np.array(lattice.get_cartesian_coords(1))
        logging.debug("In atomic units, lat consts are (in bohr):" + str([a1, a2, a3]))
        vol = np.dot(a1, np.cross(a2, a3))  # vol in bohr^3

        def e_iso(encut):
            gcut = eV_to_k(encut)  # gcut is in units of 1/A
            return scipy.integrate.quad(lambda g: self.q_model.rho_rec(g * g) ** 2, step, gcut)[0] * (q**2) / np.pi

        def e_per(encut):
            eper = 0
            for g2 in generate_reciprocal_vectors_squared(a1, a2, a3, encut):
                eper += (self.q_model.rho_rec(g2) ** 2) / g2
            eper *= (q**2) * 2 * round(np.pi, 6) / vol
            eper += (q**2) * 4 * round(np.pi, 6) * self.q_model.rho_rec_limit0 / vol
            return eper

        eiso = converge(e_iso, 5, self.mad_tol, self.energy_cutoff)
        _logger.debug("Eisolated : %f", round(eiso, 5))

        eper = converge(e_per, 5, self.mad_tol, self.energy_cutoff)

        _logger.info("Eperiodic : %f hartree", round(eper, 5))
        _logger.info("difference (periodic-iso) is %f hartree", round(eper - eiso, 6))
        _logger.info("difference in (eV) is %f", round((eper - eiso) * hart_to_ev, 4))

        es_corr = round((eiso - eper) / self.dielectric * hart_to_ev, 6)
        _logger.info("Defect Correction without alignment %f (eV): ", es_corr)
        return es_corr

    def perform_pot_corr(
        self,
        axis_grid,
        pureavg,
        defavg,
        lattice,
        q,
        defect_frac_position,
        axis,
        widthsample=1.0,
    ):
        """For performing planar averaging potential alignment.

        Args:
            axis_grid (1 x NGX where NGX is the length of the NGX grid
                in the axis direction. Same length as pureavg list):
                    A numpy array which contain the Cartesian axis
                    values (in angstroms) that correspond to each planar avg
                    potential supplied.
            pureavg (1 x NGX where NGX is the length of the NGX grid in
                the axis direction.):
                    A numpy array for the planar averaged
                    electrostatic potential of the bulk supercell.
            defavg (1 x NGX where NGX is the length of the NGX grid in
                the axis direction.):
                A numpy array for the planar averaged
                electrostatic potential of the defect supercell.
            lattice: Pymatgen Lattice object of the defect supercell
                q (float or int): charge of the defect
                defect_frac_position: Fracitional Coordinates of the defect in the supercell
                axis (int): axis for performing the freysoldt correction on
                widthsample (float): width (in Angstroms) of the region in between defects
                where the potential alignment correction is averaged. Default is 1 Angstrom.

        Returns:
            Potential Alignment contribution to Freysoldt Correction (float)
        """
        logging.debug("run Freysoldt potential alignment method for axis " + str(axis))
        nx = len(axis_grid)

        # shift these planar averages to have defect at origin
        axfracval = defect_frac_position[axis]
        axbulkval = axfracval * lattice.abc[axis]
        if axbulkval < 0:
            axbulkval += lattice.abc[axis]
        elif axbulkval > lattice.abc[axis]:
            axbulkval -= lattice.abc[axis]

        if axbulkval:
            for i in range(nx):
                if axbulkval < axis_grid[i]:
                    break
            rollind = len(axis_grid) - i
            pureavg = np.roll(pureavg, rollind)
            defavg = np.roll(defavg, rollind)

        # if not self._silence:
        _logger.debug("calculating lr part along planar avg axis")
        reci_latt = lattice.reciprocal_lattice
        dg = reci_latt.abc[axis]
        dg /= ang_to_bohr  # convert to bohr to do calculation in atomic units

        # Build background charge potential with defect at origin
        v_G = np.empty(len(axis_grid), np.dtype("c16"))
        v_G[0] = 4 * np.pi * -q / self.dielectric * self.q_model.rho_rec_limit0
        g = np.roll(np.arange(-nx / 2, nx / 2, 1, dtype=int), int(nx / 2)) * dg
        g2 = np.multiply(g, g)[1:]
        v_G[1:] = 4 * np.pi / (self.dielectric * g2) * -q * self.q_model.rho_rec(g2)
        v_G[nx // 2] = 0 if not (nx % 2) else v_G[nx // 2]

        # Get the real space potential by performing a  fft and grabbing the imaginary portion
        v_R = np.fft.fft(v_G)

        if abs(np.imag(v_R).max()) > self.mad_tol:
            raise Exception("imaginary part found to be %s", repr(np.imag(v_R).max()))
        v_R /= lattice.volume * ang_to_bohr**3
        v_R = np.real(v_R) * hart_to_ev

        # get correction
        short = np.array(defavg) - np.array(pureavg) - np.array(v_R)
        checkdis = int((widthsample / 2) / (axis_grid[1] - axis_grid[0]))
        mid = int(len(short) / 2)

        tmppot = [short[i] for i in range(mid - checkdis, mid + checkdis + 1)]
        _logger.debug("shifted defect position on axis (%s) to origin", repr(axbulkval))
        _logger.debug(
            "means sampling region is (%f,%f)",
            axis_grid[mid - checkdis],
            axis_grid[mid + checkdis],
        )

        C = -np.mean(tmppot)
        _logger.debug("C = %f", C)
        final_shift = [short[j] + C for j in range(len(v_R))]
        v_R = [elmnt - C for elmnt in v_R]

        _logger.info("C value is averaged to be %f eV ", C)
        _logger.info("Potentital alignment energy correction (-q*delta V):  %f (eV)", -q * C)
        self.pot_corr = -q * C

        # log plotting data:
        self.metadata["pot_plot_data"][axis] = {
            "Vr": v_R,
            "x": axis_grid,
            "dft_diff": np.array(defavg) - np.array(pureavg),
            "final_shift": final_shift,
            "check": [mid - checkdis, mid + checkdis + 1],
        }

        # log uncertainty:
        self.metadata["pot_corr_uncertainty_md"][axis] = {
            "stats": stats.describe(tmppot)._asdict(),
            "potcorr": -q * C,
        }

        return self.pot_corr

    def plot(self, axis, title=None, saved=False):
        """Plot the planar average electrostatic potential.

        Plot the planar average electrostatic potential against the Long range and
        short range models from Freysoldt. Must run perform_pot_corr or get_correction
        (to load metadata) before this can be used.

        Args:
            axis (int): axis to plot
            title (str): Title to be given to plot. Default is no title.
            saved (bool): Whether to save file or not. If False then returns plot object.
            If True then saves plot as   str(title) + "FreyplnravgPlot.pdf"
        """
        if not self.metadata["pot_plot_data"]:
            raise ValueError("Cannot plot potential alignment before running correction!")

        x = self.metadata["pot_plot_data"][axis]["x"]
        v_R = self.metadata["pot_plot_data"][axis]["Vr"]
        dft_diff = self.metadata["pot_plot_data"][axis]["dft_diff"]
        final_shift = self.metadata["pot_plot_data"][axis]["final_shift"]
        check = self.metadata["pot_plot_data"][axis]["check"]

        plt.figure()
        plt.clf()
        plt.plot(x, v_R, c="green", zorder=1, label="long range from model")
        plt.plot(x, dft_diff, c="red", label="DFT locpot diff")
        plt.plot(x, final_shift, c="blue", label="short range (aligned)")

        tmpx = [x[i] for i in range(check[0], check[1])]
        plt.fill_between(tmpx, -100, 100, facecolor="red", alpha=0.15, label="sampling region")

        plt.xlim(round(x[0]), round(x[-1]))
        ymin = min(min(v_R), min(dft_diff), min(final_shift))
        ymax = max(max(v_R), max(dft_diff), max(final_shift))
        plt.ylim(-0.2 + ymin, 0.2 + ymax)
        plt.xlabel(r"distance along axis ($\AA$)", fontsize=15)
        plt.ylabel("Potential (V)", fontsize=15)
        plt.legend(loc=9)
        plt.axhline(y=0, linewidth=0.2, color="black")
        plt.title(str(title) + " defect potential", fontsize=18)
        plt.xlim(0, max(x))
        if saved:
            plt.savefig(str(title) + "FreyplnravgPlot.pdf")
            return None
        return plt

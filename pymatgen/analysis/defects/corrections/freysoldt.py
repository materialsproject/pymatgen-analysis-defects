"""Freysoldt defect corrections module."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import scipy
from pymatgen.analysis.defects.finder import DefectSiteFinder
from pymatgen.analysis.defects.utils import (
    CorrectionResult,
    QModel,
    ang_to_bohr,
    converge,
    eV_to_k,
    genrecip,
    hart_to_ev,
)
from pymatgen.io.vasp.outputs import Locpot, VolumetricData
from scipy import stats
from scipy.special import elliprf

_logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from numpy.typing import ArrayLike
    from pymatgen.core import Lattice

__author__ = "Jimmy-Xuan Shen, Danny Broberg, Shyam Dwaraknath"
__copyright__ = "Copyright 2022, The Materials Project"
__maintainer__ = "Jimmy-Xuan Shen"
__email__ = "jmmshn@gmail.com"

_logger = logging.getLogger(__name__)


"""
Adapted from the original code by Danny and Shyam.
Rewritten to be functional instead of object oriented.
"""


def _get_dielectric_tensor(dielectric: float | ArrayLike) -> np.ndarray:
    """Convert a scalar, 3-vector (diagonal) or 3x3 dielectric to a 3x3 tensor."""
    dielectric = np.asarray(dielectric, dtype=float)
    if dielectric.shape in {(), (3,)}:
        return np.diag(np.broadcast_to(dielectric, (3,)))
    if dielectric.shape == (3, 3):
        return dielectric
    msg = f"Dielectric constant must be a scalar, 3-vector or 3x3 matrix, but got shape {dielectric.shape}"
    raise ValueError(msg)


def get_spherically_averaged_dielectric(dielectric: float | ArrayLike) -> float:
    r"""Get the spherically-averaged (isotropic) dielectric constant of a dielectric tensor.

    Defined as :math:`1/\bar{\epsilon} = \langle 1/(\hat{g}^T \epsilon \hat{g}) \rangle`,
    averaged over unit vectors :math:`\hat{g}`, as used by ``sxdefectalign`` for the
    isolated energy and potential alignment (G=0) terms with anisotropic screening.
    This is exact for the spherically-averaged potential of a point charge
    (:math:`q/\bar{\epsilon}r`), and so for the self-energy of a spherical model
    charge. It lies between the harmonic and arithmetic means of the eigenvalues
    :math:`\lambda_i`, and is evaluated as
    :math:`\sqrt{\det\epsilon} / R_F(1/\lambda_1, 1/\lambda_2, 1/\lambda_3)`, with
    :math:`R_F` the Carlson symmetric elliptic integral.

    Args:
        dielectric: Dielectric constant (scalar), diagonal (3-vector) or 3x3 tensor.

    Returns:
        float: The spherically-averaged dielectric constant.
    """
    dielectric_tensor = _get_dielectric_tensor(dielectric)
    if np.array_equal(dielectric_tensor, dielectric_tensor[0, 0] * np.eye(3)):
        return float(dielectric_tensor[0, 0])  # isotropic
    eigvals = np.linalg.eigvalsh((dielectric_tensor + dielectric_tensor.T) / 2)
    return float(np.sqrt(np.prod(eigvals)) / elliprf(*(1 / eigvals)))


def get_freysoldt_correction(
    q: int,
    dielectric: float | ArrayLike,
    defect_locpot: Locpot,
    bulk_locpot: Locpot,
    defect_frac_coords: ArrayLike | None = None,
    lattice: Lattice | None = None,
    energy_cutoff: float = 520,
    mad_tol: float = 1e-4,
    q_model: QModel | None = None,
) -> CorrectionResult:
    """Gets the Freysoldt correction for a defect entry.

    Anisotropic dielectric tensors are treated as in ``sxdefectalign``: the full tensor
    is used for the periodic energy and the (planar-averaged) long-range model
    potential along each axis, with the spherically-averaged dielectric constant
    (:func:`get_spherically_averaged_dielectric`) used for the isolated energy and
    the G=0 (alignment) term of the model potential.

    Args:
        q:
            Charge state of defect
        dielectric:
            Dielectric constant of bulk
        defect_locpot:
            Locpot of defect
        bulk_locpot:
            Locpot of bulk
        defect_frac_coords:
            Fractional coordinates of the defect.
        lattice:
            Lattice of the defect supercell. If None, then uses the lattice of the
            defect_locpot.
        energy_cutoff:
            Maximum energy in eV in reciprocal space to perform integration
        mad_tol:
            Convergence criteria for the Madelung energy for potential correction
        q_model:
            QModel object to use for Freysoldt correction. If None, then uses default

    Returns:
        CorrectionResult: Correction summary object. The metadata contains
            plotting data for the planar average electrostatic potential.
            ```
            plot_plnr_avg(result.metadata[0], title="Lattice Direction 1")
            ```
    """
    if defect_frac_coords is None:
        finder = DefectSiteFinder()
        defect_frac_coords = finder.get_defect_fpos(
            defect_structure=defect_locpot.structure,
            base_structure=bulk_locpot.structure,
        )

    dielectric = _get_dielectric_tensor(dielectric)
    q_model = QModel() if q_model is None else q_model

    if isinstance(defect_locpot, VolumetricData):
        list_axis_grid = [*map(defect_locpot.get_axis_grid, [0, 1, 2])]
        list_defect_plnr_avg_esp = [
            *map(defect_locpot.get_average_along_axis, [0, 1, 2]),
        ]
        lattice_ = defect_locpot.structure.lattice.copy()
        if lattice is not None and lattice != lattice_:
            msg = "Lattice of defect_locpot and user provided lattice do not match."
            raise ValueError(
                msg,
            )
        lattice = lattice_
    elif isinstance(defect_locpot, dict):
        defect_locpot_ = {int(k): v for k, v in defect_locpot.items()}
        list_defect_plnr_avg_esp = [defect_locpot_[i] for i in range(3)]
        list_axis_grid = [
            *map(
                np.linspace,
                [0, 0, 0],
                lattice.abc,
                [len(i) for i in list_defect_plnr_avg_esp],
            ),
        ]
    else:
        msg = "defect_locpot must be of type Locpot or dict"
        raise ValueError(msg)

    if isinstance(bulk_locpot, VolumetricData):
        list_bulk_plnr_avg_esp = [*map(bulk_locpot.get_average_along_axis, [0, 1, 2])]
    elif isinstance(bulk_locpot, dict):
        bulk_locpot_ = {int(k): v for k, v in bulk_locpot.items()}
        list_bulk_plnr_avg_esp = [bulk_locpot_[i] for i in range(3)]
    else:  # pragma: no cover
        msg = "bulk_locpot must be of type Locpot or dict"
        raise ValueError(msg)

    es_corr = perform_es_corr(
        lattice=lattice,
        q=q,
        dielectric=dielectric,
        q_model=q_model,
        energy_cutoff=energy_cutoff,
        mad_tol=mad_tol,
    )

    alignment_corrs = {}
    plot_data = {}

    for x, pureavg, defavg, axis in zip(
        list_axis_grid,
        list_bulk_plnr_avg_esp,
        list_defect_plnr_avg_esp,
        [0, 1, 2],
        strict=False,
    ):
        alignment_corr, md = perform_pot_corr(
            axis_grid=x,
            pureavg=pureavg,
            defavg=defavg,
            lattice=lattice,
            q=q,
            defect_frac_coords=defect_frac_coords,
            axis=axis,
            dielectric=dielectric,
            q_model=q_model,
            mad_tol=mad_tol,
            widthsample=1.0,
        )
        alignment_corrs[axis] = alignment_corr
        plot_data[axis] = md

    mean_alignment = np.mean(list(alignment_corrs.values()))
    pot_corr = mean_alignment * q

    return CorrectionResult(
        correction_energy=es_corr + pot_corr,
        metadata={
            "plot_data": plot_data,
            "electrostatic": es_corr,
            "alignments": alignment_corrs,
            "mean_alignments": mean_alignment,
            "potential": pot_corr,
        },
    )


def perform_es_corr(
    lattice: Lattice,
    q: float,
    dielectric: float | ArrayLike,
    q_model: QModel,
    energy_cutoff: float = 520,
    mad_tol: float = 1e-4,
) -> float:
    r"""Perform Electrostatic Freysoldt Correction.

    Perform the electrostatic Freysoldt correction for a defect. For anisotropic
    dielectric tensors, the periodic energy is screened by the full tensor
    (:math:`G^T \epsilon G`), while the isolated energy and G=0 term are screened
    by the spherically-averaged dielectric constant, as in ``sxdefectalign``.

    Args:
        lattice: Pymatgen lattice object
        q: Charge of defect
        dielectric: Dielectric constant of bulk
        q_model: QModel object to use for Freysoldt correction. If None, uses default
        energy_cutoff: Maximum energy in eV in reciprocal space to perform integration
        mad_tol: Convergence criteria for the Madelung energy for potential correction

    Return:
        float:
            Electrostatic Point Charge contribution to Freysoldt Correction (float)
    """
    _logger.info(
        "Running Freysoldt 2011 PC calculation (should be equivalent to sxdefectalign)",
    )
    _logger.debug("defect lattice constants are (in angstroms) %s", str(lattice.abc))

    [a1, a2, a3] = ang_to_bohr * np.array(lattice.get_cartesian_coords(1))
    _logger.debug("In atomic units, lat consts are (in bohr): %s", str([a1, a2, a3]))
    vol = np.dot(a1, np.cross(a2, a3))  # vol in bohr^3
    dielectric_tensor = _get_dielectric_tensor(dielectric)
    avg_eps = get_spherically_averaged_dielectric(dielectric_tensor)

    def e_iso(encut: float) -> float:
        gcut = eV_to_k(encut)  # gcut is in units of 1/A
        return (
            scipy.integrate.quad(lambda g: q_model.rho_rec(g * g) ** 2, 0, gcut)[0]
            * (q**2)
            / np.pi
        )

    def e_per(encut: float) -> float:
        eper = 0.0
        for g in genrecip(
            a1, a2, a3, encut
        ):  # times avg_eps; divided by avg_eps below:
            eper += avg_eps * q_model.rho_rec(g @ g) ** 2 / (g @ dielectric_tensor @ g)
        eper *= (q**2) * 2 * round(np.pi, 6) / vol
        eper += (q**2) * 4 * round(np.pi, 6) * q_model.rho_rec_limit0 / vol
        return eper

    eiso = converge(e_iso, 5, mad_tol, energy_cutoff)
    _logger.debug("Eisolated : %f", round(eiso, 5))

    eper = converge(e_per, 5, mad_tol, energy_cutoff)

    _logger.info("Eperiodic : %f hartree", round(eper, 5))
    _logger.info("difference (periodic-iso) is %f hartree", round(eper - eiso, 6))
    _logger.info("difference in (eV) is %f", round((eper - eiso) * hart_to_ev, 4))

    es_corr = round((eiso - eper) / avg_eps * hart_to_ev, 6)
    _logger.info("Defect Correction without alignment %f (eV): ", es_corr)
    return es_corr


def perform_pot_corr(
    axis_grid: ArrayLike,
    pureavg: ArrayLike,
    defavg: ArrayLike,
    lattice: Lattice,
    q: float,
    defect_frac_coords: ArrayLike,
    axis: Axes,
    dielectric: float | ArrayLike,
    q_model: QModel,
    mad_tol: float = 1e-4,
    widthsample: float = 1.0,
) -> tuple[float, dict]:
    r"""For performing planar averaging potential alignment.

    For anisotropic dielectric tensors, the planar-averaged long-range model
    potential along the axis is screened by the effective dielectric constant
    :math:`\hat{b}^T \epsilon \hat{b}` along the corresponding reciprocal lattice
    vector :math:`\hat{b}` (the only G-vectors contributing to the planar average),
    with the G=0 term screened by the spherically-averaged dielectric constant, as in
    ``sxdefectalign``.

    Args:
        axis_grid:
            (1 x NGX where NGX is the length of the NGX grid
            in the axis direction. Same length as pureavg list)
            A numpy array which contain the Cartesian axis
            values (in angstroms) that correspond to each planar avg
            potential supplied.
        pureavg:
            (1 x NGX where NGX is the length of the NGX grid in
            the axis direction.)
            A numpy array for the planar averaged
            electrostatic potential of the bulk supercell.
        defavg:
            (1 x NGX where NGX is the length of the NGX grid in
            the axis direction.):
            A numpy array for the planar averaged
            electrostatic potential of the defect supercell.
        lattice: Pymatgen Lattice object of the defect supercell
        q (float or int): charge of the defect
        defect_frac_coords: Fractional Coordinates of the defect in the supercell
        axis (int): axis for performing the Freysoldt correction on.
        dielectric (float or ArrayLike): dielectric constant of the bulk
        q_model (QModel): QModel object to use for Freysoldt correction. If None, uses default.
        mad_tol (float): convergence criteria for the Madelung energy for potential correction.
        widthsample (float): width (in Angstroms) of the region in between defects
            where the potential alignment correction is averaged. Default is 1 Angstrom.

    Returns:
        (float) Potential Alignment shift required to make the short range potential
        zero far from the defect.  (-C) in the Freysoldt paper.
    """
    _logger.debug("run Freysoldt potential alignment method for axis %s", str(axis))
    nx = len(axis_grid)

    # shift these planar averages to have defect at origin
    axfracval = defect_frac_coords[axis]
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

    dielectric_tensor = _get_dielectric_tensor(dielectric)
    avg_eps = get_spherically_averaged_dielectric(dielectric_tensor)
    # only G parallel to the reciprocal lattice vector contribute to the planar average:
    b_vec = reci_latt.matrix[axis]
    eps_axis = float(b_vec @ dielectric_tensor @ b_vec / (b_vec @ b_vec))

    # Build background charge potential with defect at origin
    v_G = np.empty(len(axis_grid), np.dtype("c16"))
    v_G[0] = 4 * np.pi * -q / avg_eps * q_model.rho_rec_limit0
    g = np.roll(np.arange(-nx // 2, nx // 2, 1, dtype=int), int(nx // 2)) * dg
    g2 = np.multiply(g, g)[1:]
    v_G[1:] = 4 * np.pi / (eps_axis * g2) * -q * q_model.rho_rec(g2)
    v_G[nx // 2] = 0 if not (nx % 2) else v_G[nx // 2]

    # Get the real space potential via fft and grabbing the imaginary portion
    v_R = np.fft.fft(v_G)

    if abs(np.imag(v_R).max()) > mad_tol:
        msg = "imaginary part found to be %s"
        raise Exception(msg, repr(np.imag(v_R).max()))
    v_R /= lattice.volume * ang_to_bohr**3
    v_R = np.real(v_R) * hart_to_ev

    # get correction
    short = (
        np.array(defavg, dtype=float)
        - np.array(pureavg, dtype=float)
        - np.array(v_R, dtype=float)
    )
    checkdis = int((widthsample / 2) / (axis_grid[1] - axis_grid[0]))
    mid = int(len(short) / 2)

    tmppot = [short[i] for i in range(mid - checkdis, mid + checkdis + 1)]
    _logger.debug("shifted defect position on axis (%s) to origin", repr(axbulkval))
    _logger.debug(
        "means sampling region is (%f,%f)",
        axis_grid[mid - checkdis],
        axis_grid[mid + checkdis],
    )

    C = np.mean(tmppot)
    _logger.debug("C = %f", C)
    short_range = short
    v_R = list(v_R)

    _logger.info("C value is averaged to be %f eV ", C)
    _logger.info("Potentital alignment energy correction (q * Delta):  %f (eV)", q * C)

    # log plotting data:
    metadata = {}
    metadata["pot_plot_data"] = {
        "Vr": v_R,
        "x": axis_grid,
        "dft_diff": np.array(defavg) - np.array(pureavg),
        "short_range": short_range,
        "shift": C,
        "check": [mid - checkdis, mid + checkdis + 1],
    }
    # log uncertainty:
    metadata["pot_corr_uncertainty_md"] = {
        "stats": stats.describe(tmppot)._asdict(),
        "potcorr": C,
    }
    return C, metadata


def plot_plnr_avg(
    plot_data: dict, title: str | None = None, saved: bool = False, ax: Axes = None
) -> Axes:
    """Plot the planar average electrostatic potential.

    Plot the planar average electrostatic potential against the Long range and
    short range models from Freysoldt. Must run perform_pot_corr or get_correction
    (to load metadata) before this can be used.

    Args:
        plot_data (dict): Dictionary of FreysoldtCorrection metadata.
        title (str): Title to be given to plot. Default is no title.
        saved (bool): Whether to save file or not. If False then returns plot object.
            If True then saves plot as   str(title) + "FreyplnravgPlot.pdf"
        ax (matplotlib.axes.Axes): Axes object to plot on. If None, makes new figure.
    """
    if not plot_data["pot_plot_data"]:  # pragma: no cover
        msg = "Cannot plot potential alignment before running correction!"
        raise ValueError(msg)

    x = plot_data["pot_plot_data"]["x"]
    v_R = plot_data["pot_plot_data"]["Vr"]
    dft_diff = plot_data["pot_plot_data"]["dft_diff"]
    short_range = plot_data["pot_plot_data"]["short_range"]
    check = plot_data["pot_plot_data"]["check"]
    C = plot_data["pot_plot_data"]["shift"]

    if ax is None:
        fig, ax = plt.subplots()
    ax.plot(x, v_R, c="black", zorder=1, label="long range from model")
    ax.plot(x, dft_diff, c="red", label="DFT locpot diff")
    ax.plot(x, short_range, c="green", label="short range (not aligned)")
    ax.axhline(C, color="k", linestyle="--")
    ax.text(4, C, f"C={C:0.3f}", va="bottom")

    tmpx = [x[i] for i in range(check[0], check[1])]
    ax.fill_between(
        tmpx,
        -100,
        100,
        facecolor="red",
        alpha=0.15,
        label="sampling region",
    )

    ax.set_xlim(round(x[0]), round(x[-1]))
    ymin = min(v_R + dft_diff + short_range)
    ymax = max(v_R + dft_diff + short_range)
    ax.set_ylim(-0.2 + ymin, 0.2 + ymax)
    ax.set_xlabel(r"distance along axis ($\AA$)", fontsize=15)
    ax.set_ylabel("Potential (V)", fontsize=15)
    ax.legend(loc=9)
    ax.axhline(y=0, linewidth=0.2, color="black")
    if title is not None:  # pragma: no cover
        ax.set_title(str(title), fontsize=18)
    ax.set_xlim(0, max(x))
    if saved:  # pragma: no cover
        fig.savefig(str(title) + "FreyplnravgPlot.pdf")
    return ax

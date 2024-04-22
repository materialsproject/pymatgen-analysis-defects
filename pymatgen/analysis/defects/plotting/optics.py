"""Plotting functions."""

from __future__ import annotations

import collections
import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import Colormap, Normalize
from pymatgen.electronic_structure.core import Spin

if TYPE_CHECKING:
    from collections.abc import Sequence

    from matplotlib.axes import Axes
    from pymatgen.analysis.defects.ccd import HarmonicDefect

__author__ = "Jimmy Shen"
__copyright__ = "Copyright 2022, The Materials Project"
__maintainer__ = "Jimmy Shen @jmmshn"
__date__ = "July 2023"

logger = logging.getLogger(__name__)


def plot_optical_transitions(
    defect: HarmonicDefect,
    kpt_index: int = 0,
    band_window: int = 5,
    user_defect_band: tuple = (),
    other_defect_bands: list[int] | None = None,
    ijdirs: Sequence[tuple] | None = None,
    shift_eig: dict[tuple, float] | None = None,
    x0: float = 0,
    x_width: float = 2,
    ax: Axes = None,
    cmap: Colormap = None,
    norm: Normalize = None,
) -> tuple[pd.DataFrame, Colormap, Normalize]:
    """Plot the optical transitions from the defect state to all other states.

    Only plot the transitions for a specific kpoint index. The arrows present the transitions
    between the defect state of interest and all other states. The color of the arrows
    indicate the magnitude of the matrix element (derivative of the wavefunction) for the
    transition.

    Args:
        defect:
            The HarmonicDefect object, the `relaxed_bandstructure` attribute
            must be set since this contains the eigenvalues.
            Please see the `store_bandstructure` option in the constructor.
        kpt_index:
            The kpoint index to read the eigenvalues from.
        band_window:
            The number of bands above and below the defect state to include in the output.
        user_defect_band:
            (band, kpt, spin) tuple to specify the defect state. If not provided,
            the defect state will be determined automatically using the inverse
            participation ratio and the `kpt_index` argument.
        other_defect_bands:
            A list of band indices to exclude from the plot. This is useful for
        ijdirs:
            The cartesian direction of the WAVDER tensor to sum over for the plot.
            If not provided, all the absolute values of the matrix for all
            three diagonal entries will be summed.
        shift_eig:
            A dictionary of the format `(band, kpt, spin) -> float` to apply to the
            eigenvalues. This is useful for aligning the defect state with the
            valence or conduction band for plotting and schematic purposes.
        x0:
            The x coordinate of the center of the set of arrows and the eigenvalue plot.
        x_width:
            The width of the set of arrows and the eigenvalue plot.
        ax:
            The matplotlib axis object to plot on.
        cmap:
            The matplotlib color map to use for the color of the arrorws.
        norm:
            The matplotlib normalization to use for the color map of the arrows.

    Returns:
        A pandas dataframe with the following columns:
            ib: The band index of the state the arrow is pointing to.
            jb: The band index of the defect state.
            kpt: The kpoint index of the state the arrow is pointing to.
            spin: The spin index of the state the arrow is pointing to.
            eig: The eigenvalue of the state the arrow is pointing to.
            M.E.: The matrix element of the transition.
        cmap:
            The matplotlib color map used.
        norm:
            The matplotlib normalization used.
    """
    d_eigs = get_bs_eigenvalues(
        defect=defect,
        kpt_index=kpt_index,
        band_window=band_window,
        user_defect_band=user_defect_band,
        other_defect_bands=other_defect_bands,
        shift_eig=shift_eig,
    )
    if user_defect_band:
        defect_band_index = user_defect_band[0]
    else:
        defect_band_index = next(
            filter(lambda x: x[1] == kpt_index, defect.defect_band),
        )[0]
    ax_ = plt.gca() if ax is None else ax
    _plot_eigs(
        d_eigs,
        defect.relaxed_bandstructure.efermi,
        ax=ax_,
        x0=x0,
        x_width=x_width,
    )
    ijdirs = ijdirs or ((0, 0), (1, 1), (2, 2))
    me_plot_data, cmap, norm = _plot_matrix_elements(
        defect.waveder.cder,
        d_eigs,
        defect_band_index=defect_band_index,
        ijdirs=ijdirs,
        ax=ax_,
        x0=x0,
        x_width=x_width,
        cmap=cmap,
        norm=norm,
    )
    return _get_dataframe(d_eigs=d_eigs, me_plot_data=me_plot_data), cmap, norm


def get_bs_eigenvalues(
    defect: HarmonicDefect,
    kpt_index: int = 0,
    band_window: int = 5,
    user_defect_band: tuple | None = None,
    other_defect_bands: list[int] | None = None,
    shift_eig: dict[tuple, float] | None = None,
) -> dict[tuple, float]:
    """Read the eigenvalues from `HarmonicDefect.relaxed_bandstructure`.

    Args:
        defect:
            The HarmonicDefect object, the `relaxed_bandstructure` attribute
            must be set since this contains the eigenvalues.
            Please see the `store_bandstructure` option in the constructor.
        kpt_index:
            The kpoint index to read the eigenvalues from.
        band_window:
            The number of bands above and below the Fermi level to include.
        user_defect_band:
            (band, kpt, spin) tuple to specify the defect state. If not provided,
            the defect state will be determined automatically using the inverse
            participation ratio.
            The user provided kpoint index here will overwrite the kpt_index argument.
        other_defect_bands:
            A list of band indices to exclude from the plot.
        shift_eig:
            A dictionary of the format `(band, kpt, spin) -> float` to apply to the
            eigenvalues. This is useful for aligning the defect state with the
            valence or conduction band for plotting and schematic purposes.


    Returns:
        Dictionary of the format: (iband, ikpt, ispin) -> eigenvalue
    """
    if defect.relaxed_bandstructure is None:  # pragma: no cover
        msg = "The defect object does not have a band structure."
        raise ValueError(msg)

    other_defect_bands = other_defect_bands or []

    if user_defect_band:
        def_indices = user_defect_band
    else:
        def_indices = next(filter(lambda x: x[1] == kpt_index, defect.defect_band))

    band_index, kpt_index, spin_index = def_indices
    spin_key = Spin.up if spin_index == 0 else Spin.down
    output: dict[tuple, float] = {}
    shift_dict: dict = collections.defaultdict(lambda: 0.0)
    if shift_eig is not None:
        shift_dict.update(shift_eig)
    for ib in range(band_index - band_window, band_index + band_window + 1):
        if ib in other_defect_bands:
            continue
        output[(ib, kpt_index, spin_index)] = (
            defect.relaxed_bandstructure.bands[spin_key][ib, kpt_index]
            + shift_dict[(ib, kpt_index, spin_index)]
        )
    return output


def _plot_eigs(
    d_eigs: dict[tuple, float],
    e_fermi: float | None = None,
    ax: Axes = None,
    x0: float = 0.0,
    x_width: float = 0.3,
    **kwargs,
) -> None:
    """Plot the eigenvalues.

    Args:
        d_eigs:
            The dictionary of eigenvalues for the defect state. In the format of
            (iband, ikpt, ispin) -> eigenvalue
        e_fermi:
            The bands above and below the Fermi level will be colored differently.
            If not provided, they will all be colored the same.
        ax:
            The matplotlib axis object to plot on.
        x0:
            The x coordinate of the center of the set of lines representing the eigenvalues.
        x_width:
            The width of the set of lines representing the eigenvalues.
        **kwargs:
            Keyword arguments to pass to `matplotlib.pyplot.hlines`.
            For example, `linestyles`, `alpha`, etc.
    """
    if ax is None:  # pragma: no cover
        ax = plt.gca()

    # Use current color scheme
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    collections.defaultdict(list)
    eigenvalues = np.array(list(d_eigs.values()))
    if e_fermi is None:  # pragma: no cover
        e_fermi = -np.inf

    eigs_ = eigenvalues[eigenvalues <= e_fermi]
    ax.hlines(
        eigs_,
        x0 - (x_width / 2.0),
        x0 + (x_width / 2.0),
        color=colors[0],
        **kwargs,
    )
    eigs_ = eigenvalues[eigenvalues > e_fermi]
    ax.hlines(
        eigs_,
        x0 - (x_width / 2.0),
        x0 + (x_width / 2.0),
        color=colors[1],
        **kwargs,
    )


def _plot_matrix_elements(
    cder: dict[tuple, float],
    d_eig: dict[tuple, float],
    defect_band_index: int,
    ijdirs: Sequence[tuple] = ((0, 0), (1, 1), (2, 2)),
    ax: Axes = None,
    x0: float = 0.0,
    x_width: float = 0.6,
    arrow_width: float = 0.1,
    cmap: Colormap = None,
    norm: Normalize = None,
) -> tuple[list[tuple], Colormap, Normalize]:
    """Plot arrow for the transition from the defect state to all other states.

    Args:
        cder:
            The matrix element (derivative of the wavefunction) for the defect state.
        d_eig:
            The dictionary of eigenvalues for the defect state. In the format of
            (iband, ikpt, ispin) -> eigenvalue
        defect_band_index:
            The band index of the defect state.
        ax:
            The matplotlib axis object to plot on.
        x0:
            The x coordinate of the center of the set of arrows.
        x_width:
            The width of the set of arrows.
        arrow_width:
            The width of the arrow.
        cmap:
            The matplotlib color map to use.
        norm:
            The matplotlib normalization to use for the color map.
        ijdirs:
            The cartesian direction of the WAVDER tensor to sum over for the plot.
            If not provided, all the absolute values of the matrix for all
            three diagonal entries will be summed.

    Returns:
        plot_data:
            A list of tuples in the format of (iband, ikpt, ispin, eigenvalue, matrix element)
        cmap:
            The matplotlib color map used.
        norm:
            The matplotlib normalization used.
    """
    if ax is None:  # pragma: no cover
        ax = plt.gca()
    ax.set_aspect("equal")
    jb, jkpt, jspin = next(filter(lambda x: x[0] == defect_band_index, d_eig.keys()))
    y0 = d_eig[jb, jkpt, jspin]
    plot_data: list[tuple] = []
    for (ib, ik, ispin), eig in d_eig.items():
        A = 0
        for idir, jdir in ijdirs:
            A += np.abs(
                cder[ib, jb, ik, ispin, idir]
                * np.conjugate(cder[ib, jb, ik, ispin, jdir]),
            )
        plot_data.append((jb, ib, eig, A))

    if cmap is None:
        cmap = plt.get_cmap("viridis")

    # get the range of A values
    if norm is None:
        A_min, A_max = (
            min(plot_data, key=lambda x: x[3])[3],
            max(plot_data, key=lambda x: x[3])[3],
        )
        norm = Normalize(vmin=A_min, vmax=A_max)

    n_arrows = len(plot_data)
    x_step = x_width / n_arrows
    x = x0 - x_width / 2 + x_step / 2
    for _ib, _jb, eig, A in plot_data:
        ax.arrow(
            x=x,
            y=y0,
            dx=0,
            dy=eig - y0,
            width=arrow_width,
            length_includes_head=True,
            head_width=arrow_width * 2,
            head_length=arrow_width * 2,
            color=cmap(norm(A)),
            zorder=20,
        )
        x += x_step
    return plot_data, cmap, norm


def _get_dataframe(d_eigs: dict, me_plot_data: list[tuple]) -> pd.DataFrame:
    """Convert the eigenvalue and matrix element data into a pandas dataframe.

    Args:
        d_eigs:
            The dictionary of eigenvalues for the defect state. In the format of
            (iband, ikpt, ispin) -> eigenvalue
        me_plot_data:
            A list of tuples in the format of (iband, ikpt, ispin, eigenvalue, matrix element)

    Returns:
        A pandas dataframe with the following columns:
            ib: The band index of the state the arrow is pointing to.
            jb: The band index of the defect state.
            kpt: The kpoint index of the state the arrow is pointing to.
            spin: The spin index of the state the arrow is pointing to.
            eig: The eigenvalue of the state the arrow is pointing to.
            M.E.: The matrix element of the transition.
    """
    _, ikpt, ispin = next(iter(d_eigs.keys()))
    output_dataframe = pd.DataFrame(
        me_plot_data,
        columns=["ib", "jb", "eig", "M.E."],
    )
    output_dataframe["kpt"] = ikpt
    output_dataframe["spin"] = ispin
    return output_dataframe

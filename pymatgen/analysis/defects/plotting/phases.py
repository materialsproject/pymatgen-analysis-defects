"""Plotting functions for competing phases."""
# %%
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from matplotlib import pyplot as plt
from matplotlib.patches import Polygon
from pymatgen.util.string import latexify
from scipy.spatial import ConvexHull

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from pymatgen.analysis.defects.thermo import FormationEnergyDiagram
    from pymatgen.core import Element

# check if labellines is installed
try:
    from labellines import labelLines
except ImportError:

    def labelLines(*args, **kwargs):
        """Dummy function if labellines is not installed."""
        pass


__author__ = "Jimmy Shen"
__copyright__ = "Copyright 2022, The Materials Project"
__maintainer__ = "Jimmy Shen @jmmshn"
__date__ = "July 2023"

logger = logging.getLogger(__name__)


def plot_chempot_2d(
    fed: FormationEnergyDiagram,
    x_element: Element,
    y_element: Element,
    ax: Axes | None = None,
    min_mu: float = -5.0,
    label_lines: bool = False,
    x_vals: list[float] | None = None,
    label_fontsize: int = 12,
):
    """Plot the chemical potential diagram for two elements.

    Args:
        fed:
            The formation energy diagram.
        x_element:
            The element to use for the x-axis.
        y_element:
            The element to use for the y-axis.
        ax:
            The matplotlib axes to plot on. If None, a new figure will be created.
        min_mu:
            The minimum chemical potential to plot.
        label_lines:
            Whether to label the lines with the competing phases. Requires Labellines to be installed.
        x_vals:
            The x position of the line labels. If None, defaults will be used.
        label_fontsize:
            The fontsize for the line labels.
    """
    PLOT_PADDING = 0.1
    ax = ax or plt.gca()
    hull2d = _convex_hull_2d(
        fed.chempot_limits,
        x_element=x_element,
        y_element=y_element,
        competing_phases=fed.competing_phases,
    )
    x_min = float("inf")
    y_min = float("inf")
    clip_path = []
    for p1, p2, phase in hull2d:
        p_txt = ", ".join(map(latexify, phase.keys()))
        ax.axline(p1, p2, label=p_txt, color="k")
        ax.scatter(p1[0], p1[1], color="k")
        x_m_ = p1[0] if p1[0] > min_mu else float("inf")
        y_m_ = p1[1] if p1[1] > min_mu else float("inf")
        x_min = min(x_min, x_m_)
        y_min = min(y_min, y_m_)
        clip_path.append(p1)

    patch = Polygon(
        clip_path,
        closed=True,
    )
    ax.add_patch(patch)

    ax.set_xlabel(rf"$\Delta\mu_{{{x_element}}}$ (eV)")
    ax.set_ylabel(rf"$\Delta\mu_{{{y_element}}}$ (eV)")
    ax.set_xlim(x_min - PLOT_PADDING, 0 + PLOT_PADDING)
    ax.set_ylim(y_min - PLOT_PADDING, 0 + PLOT_PADDING)
    if label_lines:
        labelLines(ax.get_lines(), align=False, xvals=x_vals, fontsize=label_fontsize)


def _convex_hull_2d(
    points: list[dict],
    x_element: Element,
    y_element: Element,
    competing_phases: list | None = None,
) -> list[dict]:
    """Compute the convex hull of a set of points in 2D.

    Args:
        points:
            A list of dictionaries with keys "x" and "y" and values as floats.
        x_element:
            The element to use for the x-axis.
        y_element:
            The element to use for the y-axis.
        tol:
            The tolerance for determining if two points are the same in the 2D plane.
        competing_phases:
            A list of competing phases for each point.

    Returns:
        A list of dictionaries with keys "x" and "y" that form the vertices of the
        convex hull.
    """
    if competing_phases is None:
        competing_phases = [None] * len(points)
    xy_points = [(pt[x_element], pt[y_element]) for pt in points]
    hull = ConvexHull(xy_points)
    xy_hull = [xy_points[i] for i in hull.vertices]
    pt_and_phase = []

    def _get_line_data(i1, i2):
        cp1 = competing_phases[hull.vertices[i1]]
        cp2 = competing_phases[hull.vertices[i2]]
        shared_keys = cp1.keys() & cp2.keys()
        shared_phase = {k: cp1[k] for k in shared_keys}
        return xy_hull[i1], xy_hull[i2], shared_phase

    # return all pairs of points:
    for itr in range(1, len(hull.vertices)):
        pt_and_phase.append(_get_line_data(itr - 1, itr))
    pt_and_phase.append(_get_line_data(len(hull.vertices) - 1, 0))
    return pt_and_phase

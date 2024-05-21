"""Plotting functions for defect thermo properties."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from matplotlib import pyplot as plt
from matplotlib.patches import Polygon
from pymatgen.analysis.defects.thermo import group_formation_energy_diagrams
from pymatgen.util.string import latexify
from scipy.spatial import ConvexHull

from .utils import get_line_style_and_color_sequence

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from pymatgen.analysis.defects.thermo import FormationEnergyDiagram
    from pymatgen.core import Element

# check if labellines is installed
try:
    from labellines import labelLines
except ImportError:

    def labelLines(*args, **kwargs) -> None:  # noqa: ARG001, ANN002
        """Dummy function if labellines is not installed."""


PLOTLY_COLORS = px.colors.qualitative.T10
PLOTLY_STYLES = ["solid", "dot", "dash", "longdash", "dashdot", "longdashdot"]

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pymatgen.analysis.defects.thermo import FormationEnergyDiagram


def _plot_line(
    pts: Sequence,
    fig: go.Figure,
    color: str,
    style: str,
    name: str,
    meta: dict,
) -> None:
    """Plot a sequence of x, y points as a line.

    Args:
        pts: A sequence of x, y points.
        fig: A plotly figure object.
        color: The color of the line.
        style: The style of the line.
        name: The name of the line.
        meta: A dictionary of metadata.

    Returns:
        None, modifies the fig object in place.
    """
    x_pos, y_pos = tuple(zip(*pts))
    trace_ = go.Scatter(
        x=x_pos,
        y=y_pos,
        mode="lines+markers",
        textposition="top right",
        line=dict(color=color, dash=style),
        hoverinfo="x",
        meta=meta,
        name=name,
    )
    fig.add_trace(trace_)


def _label_lines(fig: go.Figure, name: str, x_anno: float, color: str) -> None:
    """Label the lines in the figure.

    Args:
        fig: A plotly figure object.
        name: The unique identifier of the line.
        x_anno: The x-coordinate of the annotation.
        color: The color of the annotation.

    Returns:
        None, modifies the fig object in place.
    """
    for trace_ in fig.select_traces(selector={"name": name}):
        x_pos, y_pos = trace_.x, trace_.y
        y_anno = np.interp(x=x_anno, xp=x_pos, fp=y_pos)
        fig.add_annotation(
            x=x_anno,
            y=y_anno,
            text=trace_.name,
            font=dict(color=color),
            arrowcolor=color,
            bgcolor="white",
            bordercolor=color,
        )


def _label_slopes(fig: go.Figure) -> None:
    """Label the slopes of the lines in the figure.

    Only labels lines that have the meta attribute 'formation_energy'.

    Args:
        fig: A plotly figure object.
    """
    for data_ in filter(lambda x: x.meta.get("formation_energy_plot", False), fig.data):
        transitions_arr_ = np.array(tuple(zip(data_.x, data_.y)))
        diff_arr = transitions_arr_[1:] - transitions_arr_[:-1]
        slopes = tuple(
            int(slope) for slope in np.round(diff_arr[:, 1] / diff_arr[:, 0])
        )
        pos = (transitions_arr_[:-1] + transitions_arr_[1:]) / 2.0
        x_pos, y_pos = tuple(zip(*pos))
        fig.add_trace(
            go.Scatter(
                x=x_pos,
                y=y_pos,
                text=slopes,
                mode="text",
                textposition="top center",
                hoverinfo="skip",
                textfont=dict(color=data_.line.color),
                name=f"{data_.name}:slope",
                showlegend=False,
            )
        )


def plot_formation_energy_diagrams(
    feds: Sequence[FormationEnergyDiagram], chempot: dict | None = None
) -> go.Figure:
    """Plot formation energy diagrams for a sequence of formation energy diagrams.

    Args:
        feds: A sequence of formation energy diagrams.
        chempot: A dictionary of chemical potentials.

    Returns:
        A plotly figure object.
    """
    fig = go.Figure()
    plot_data = get_plot_data(feds, chempot)

    for name, data in plot_data.items():
        _plot_line(
            pts=data["fed"].get_transitions(data["chempot"]),
            fig=fig,
            name=name,
            color=data["color"],
            style=data["style"],
            meta={"formation_energy_plot": True},
        )
        _label_lines(fig=fig, name=name, x_anno=data["x_anno"], color=data["color"])

    _label_slopes(fig)

    fig.update_layout(
        title="Formation Energy Diagrams",
        xaxis_title="Fermi Level (eV)",
        yaxis_title="Formation Energy (eV)",
        template="plotly_white",
        font_family="Helvetica",
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False),
        showlegend=False,
    )
    return fig


def get_plot_data(
    feds: Sequence[FormationEnergyDiagram], chempot: dict | None = None
) -> dict:
    """Get the plot data for a sequence of formation energy diagrams.

    Args:
        feds: A sequence of formation energy diagrams.
        chempot: A dictionary of chemical potentials.

    Returns:
        A dictionary of plot data.
            - key: The unique identifier (just the unique name from group_formation_energy_diagrams).
            - value: A dictionary with the following keys:
                - fed: The formation energy diagram.
                - style: The style of the line.
                - color: The color of the line.
                - x_anno: The x-coordinate of the annotation.
                - chempot: The chemical potentials used to generate the transitions.

    """
    x_annos_ = np.linspace(0, feds[0].band_gap, len(feds) + 1, endpoint=True)
    x_annos_ += (x_annos_[1] - x_annos_[0]) / 2
    x_annos_ = x_annos_[:-1]

    # Group formation energy diagrams by unique name
    grouped_feds = list(group_formation_energy_diagrams(feds))
    plot_data = dict()
    num_feds = len(grouped_feds)
    for (name_, fed), color, x_anno in zip(
        grouped_feds,
        get_line_style_and_color_sequence(PLOTLY_COLORS, PLOTLY_STYLES),
        x_annos_,
    ):
        if chempot is None:
            cation_el_ = fed.chempot_diagram.elements[0]
            chempot_ = fed.get_chempots(rich_element=cation_el_)
        else:
            chempot_ = chempot
        plot_data[name_] = dict(
            fed=fed,
            style=color[0],
            color=color[1],
            x_anno=x_anno,
            chempot=chempot_,
        )

    if len(plot_data) != num_feds:
        msg = "Duplicate Name found in formation energy diagrams. "
        raise ValueError(
            msg,
            "This should not happen since each unique defect should have a unique Name.",
        )

    return plot_data


def plot_chempot_2d(
    fed: FormationEnergyDiagram,
    x_element: Element,
    y_element: Element,
    ax: Axes | None = None,
    min_mu: float = -5.0,
    label_lines: bool = False,
    x_vals: list[float] | None = None,
    label_fontsize: int = 12,
) -> None:
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
) -> list:
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

    def _get_line_data(i1: int, i2: int) -> tuple:
        cp1 = competing_phases[hull.vertices[i1]]
        cp2 = competing_phases[hull.vertices[i2]]
        shared_keys = cp1.keys() & cp2.keys()
        shared_phase = {k: cp1[k] for k in shared_keys}
        return xy_hull[i1], xy_hull[i2], shared_phase

    # return all pairs of points:
    pt_and_phase = [
        _get_line_data(itr - 1, itr) for itr in range(1, len(hull.vertices))
    ]
    pt_and_phase.append(_get_line_data(len(hull.vertices) - 1, 0))
    return pt_and_phase

"""Plotting functions for competing phases."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pymatgen.analysis.defects.thermo import group_formation_energy_diagrams

from .utils import get_line_color_and_style_sequence

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
    uid: str,
    x_anno: float,
) -> None:
    """Plot a sequence of x, y points as a line.

    Args:
        pts: A sequence of x, y points.
        fig: A plotly figure object.
        color: The color of the line.
        style: The style of the line.
        name: The name of the line.
        uid: The unique identifier of the line.
        x_anno: The x-coordinate of the annotation.

    Returns:
        None, modifies the fig object in place.
    """
    x_pos, y_pos = tuple(zip(*pts))
    trace_ = go.Scatter(
        x=x_pos,
        y=y_pos,
        mode="lines+markers",
        name=name,
        textposition="top right",
        line=dict(color=color, dash=style),
        hoverinfo="x",
        meta="formation_energy",
        uid=uid,
    )
    y_anno = np.interp(x=x_anno, xp=x_pos, fp=y_pos)
    fig.add_annotation(
        x=x_anno,
        y=y_anno,
        text=name,
        font=dict(color=color),
        arrowcolor=color,
        bgcolor="white",
        bordercolor=color,
    )
    fig.add_trace(trace_)


def _lable_slopes(fig: go.Figure) -> None:
    """Label the slopes of the lines in the figure.

    Only labels lines that have the meta attribute 'formation_energy'.

    Args:
        fig: A plotly figure object.
    """
    for data_ in filter(lambda x: x.meta == "formation_energy", fig.data):
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
            )
        )


def _get_name(fed: FormationEnergyDiagram) -> str:
    root, suffix = fed.defect.name.split("_")
    return rf"${{\rm {root}}}_{{\rm {suffix}}}$"


def plot_formation_energy_diagrams(
    feds: Sequence[FormationEnergyDiagram], chempot: dict | None = None
) -> go.Figure:
    """Plot formation energy diagrams for a sequence of formation energy diagrams.

    Args:
        feds: A sequence of formation energy diagrams.
        chempot: A dictionary of chemical potentials.
    """
    fig = go.Figure()
    x_annos_ = np.linspace(0, feds[0].band_gap, len(feds) + 1, endpoint=True)
    x_annos_ += (x_annos_[1] - x_annos_[0]) / 2
    x_annos_ = x_annos_[:-1]

    # use structure an defect name to get uid
    grouped_feds = group_formation_energy_diagrams(feds)

    for uid, fed, color, x_anno in zip(
        *grouped_feds,
        get_line_color_and_style_sequence(PLOTLY_COLORS, PLOTLY_STYLES),
        x_annos_,
    ):
        if chempot is None:
            cation_el_ = fed.chempot_diagram.elements[0]
            chempot_ = fed.get_chempots(rich_element=cation_el_)

        _plot_line(
            pts=fed.get_transitions(chempot_),
            fig=fig,
            color=color[1],
            style=color[0],
            name=_get_name(fed),
            x_anno=x_anno,
            uid=uid,
        )
    _lable_slopes(fig)

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

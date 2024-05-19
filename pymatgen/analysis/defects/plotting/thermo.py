"""Plotting functions for competing phases."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import plotly.express as px
import plotly.graph_objects as go

PLOTLY_COLORS = px.colors.qualitative.T10

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pymatgen.analysis.defects.thermo import FormationEnergyDiagram


def _plot_line(pts: Sequence, fig: go.Figure, color: str, name: str) -> None:
    """Plot a sequence of x, y points as a line.

    Args:
        pts: A sequence of x, y points.
        fig: A plotly figure object.
        color: The color of the line.
        name: The name of the line.

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
        line=dict(color=color, dash="solid"),
        hoverinfo="x",
        meta="formation_energy",
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

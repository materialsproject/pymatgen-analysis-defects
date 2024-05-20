"""Plotting utils."""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Generator, Sequence


def get_line_style_and_color_sequence(
    colors: Sequence, styles: Sequence
) -> Generator[tuple[str, str], None, None]:
    """Get a generator for colors and styles.

    Create an iterator that will cycle through the colors and styles.

    Args:
        colors: List of colors to use.
        styles: List of styles to use.

    Returns:
        Generator of (style, color) tuples
    """
    for style in itertools.cycle(styles):
        for color in itertools.cycle(colors):
            yield style, color

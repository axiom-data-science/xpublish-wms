"""Vector rendering helpers."""

# TODO file names need work

from typing import Any, Mapping, Tuple

import numpy as np
import matplotlib
import xarray as xr

from matplotlib import pyplot as plt # noqa
from PIL.Image import Image, fromarray

matplotlib.use('Agg')


def get_meshgrid(
    density: int, tile_width: int, tile_height: int
) -> tuple[np.ndarray[Tuple[int]], np.ndarray[Tuple[int]]]:
    """Generate indices and a meshgrid for rendering vector glyphs."""
    # For a 256x256 tile, there will be:
    # - 4x4 glyphs at density 1,
    # - 8x8 glyphs for density 2,
    # - 16x16 glyphs for density 3.
    grid_step = 64 // (2 ** (density - 1))
    x_indices = np.arange(grid_step // 2, tile_width, grid_step)
    y_indices = np.arange(grid_step // 2, tile_height, grid_step)
    return x_indices, y_indices


def setup_tile_plot(tile_width: int, tile_height: int) -> tuple[plt.Figure, plt.Axes]:
    """Setup a plot with appropriate axes for rendering a tile."""
    # Make a figure without a frame or axes, this ensures the image has the desired
    # dimensions, as well as not drawing any axes.
    fig = plt.figure(frameon=False, dpi=1, figsize=(tile_width, tile_height))
    ax = plt.Axes(fig, [0, 0, 1, 1])
    ax.set_axis_off()
    # This is extremely important. It ensures that the frame of the figure used has the same
    # dimensions and orientation as the raster tile
    ax.set_xlim(0, tile_width)
    ax.set_ylim(0, tile_height)
    fig.add_axes(ax)

    return fig, ax


def render_vector_arrows(
    fig: plt.Figure,
    ax: plt.Axes,
    render_args: Tuple[np.ndarray[Tuple[int]], np.ndarray[Tuple[int]], xr.DataArray, xr.DataArray],
    render_kwargs: Mapping[str, Any],
) -> Image:
    """Plot vector arrows using `matplotlib.quiver` and create an `Image`.
    
    Also call `matplotlib.close` at the end.
    """
    # Internally matplotlib scales the arrow width and length based on the number of arrows that
    # render, and scale. We explicitly set the width to prevent arrows from being larger on
    # tiles with fewer arrows, and we set scale=1 and units='xy' so we can very carefully set
    # the arrow lengths.
    render_args = (*np.meshgrid(*render_args[:2]), *render_args[2:])
    ax.quiver(*render_args, scale=1, units='xy', **render_kwargs)

    fig.canvas.draw()
    # Copy the matplotlib figure to an Image object
    im = fromarray(np.asarray(fig.canvas.buffer_rgba()))
    plt.close()
    return im

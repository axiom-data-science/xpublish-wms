"""Visualize vector direction."""

from typing import List
import matplotlib
import numpy as np
import xarray as xr

from matplotlib import pyplot as plt # noqa
from PIL.Image import Image, fromarray

matplotlib.use('Agg')


# For arrows, the length should be uniform so we scale all u/v components by the magnitude
LENGTH_SCALE = np.array([
    [16, 12, 8],
    [32, 32, 16],
    [64, 32, 24],
])


def visualize_direction(
    mesh: xr.DataArray,
    color: str,
    density: int,
    scale: int = 1,
) -> Image:
    """Renders a vector quiver overlay."""
    # Create a mesh of grid-points where we will draw arrows/barbs
    if density not in (1, 2, 3):
        raise ValueError(f'Invalid density value {density}')

    tile_size = mesh.shape[0]
    assert tile_size == mesh.shape[1]

    # vector glyphs are tiled initially with 4x4 per tile at density 1, then 8x8 for density 2, then
    # 16x16 for density 3.
    tile_axis = np.arange(32 // (2 ** (density - 1)), tile_size, 64 // (2 ** (density - 1)))
    tile_axis *= scale
    X, Y = np.meshgrid(tile_axis, tile_axis)

    # Create normalized vectors based on the direction every n grid points
    u = np.cos(mesh.isel(x=tile_axis, y=tile_axis).astype(np.float32))
    v = np.sin(mesh.isel(x=tile_axis, y=tile_axis).astype(np.float32))

    # TODO: mask???
    # A boolean mask that will make sure we're not drawing glyphs where there's no valid data.
    # vector_mask = (mask[Y, X] == 0)

    # Scale the length up to so it renders properly.
    u *= LENGTH_SCALE[scale - 1, density - 1]
    v *= LENGTH_SCALE[scale - 1, density - 1]

    # Make a figure without a frame or axes, this ensures the image is exactly 256x256 or whatever
    # the appropriate scale is, as well as not drawing any axes.
    fig = plt.figure(frameon=False, dpi=tile_size, figsize=(scale, scale))
    ax = plt.Axes(fig, [0, 0, 1, 1])
    ax.set_axis_off()
    # This is extremely important. It ensures that the frame of the figure used has the same
    # dimensions and orientation as the raster tile
    ax.set_xlim(0, tile_size * scale)
    ax.set_ylim(0, tile_size * scale)
    fig.add_axes(ax)

    # TODO
    # Only draw glyphs where there's valid data
    # X = X[~vector_mask]
    # Y = Y[~vector_mask]
    # u = u[~vector_mask]
    # v = v[~vector_mask]
    # mag = mag[~vector_mask]

    render_args = {}
    args = (X, Y, u, v)
    render_args['color'] = color

    # Internally matplotlib scales the arrow width and length based on the number of arrows that
    # render, and scale. We explicitly set the width to prevent arrows from being larger on
    # tiles with fewer arrows, and we set scale=1 and units='xy' so we can very carefully set
    # the arrow lengths.
    width = [8, 4, 2]
    ax.quiver(*args, scale=1, units='xy', width=width[density - 1], **render_args)

    fig.canvas.draw()
    # Copy the matplotlib figure to an Image object
    im = fromarray(np.asarray(fig.canvas.buffer_rgba()))
    plt.close()
    return im


def get_colormaps() -> List[str]:
    """Returns a listing of available colormaps from matplotlib to be available for vectors."""
    return plt.colormaps()


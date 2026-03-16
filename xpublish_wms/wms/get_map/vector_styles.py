"""Visualize vector direction."""

from typing import List, Sequence
import numpy as np
import xarray as xr

from matplotlib import pyplot as plt # noqa
from PIL.Image import Image

from xpublish_wms.wms.get_map.style_types import VectorStyleParams
from xpublish_wms.wms.get_map.vectors import get_meshgrid, render_vector_arrows, setup_tile_plot



# For arrows, the length should be uniform so we scale all u/v components by the magnitude
LENGTH_SCALE = np.array([16, 12, 8])
WIDTH = [8, 4, 2]


def visualize_vectors(
    meshes: Sequence[xr.DataArray],
    color: str,
    density: int,
    scaling: VectorStyleParams.GlyphScaling,
    colorscale_range: tuple[float, float] | None = None,
    colormap: str | None = None,
) -> Image:
    """Renders a vector quiver overlay."""
    # Create a mesh of grid-points where we will draw arrows/barbs
    if density not in (1, 2, 3):
        raise ValueError(f'Invalid density value {density}')
    
    # TODO during request validation make sure that vectors visualization has two layers
    assert meshes[0].shape == meshes[1].shape
    tile_width, tile_height = meshes[0].shape

    x_indices, y_indices = get_meshgrid(density, tile_width, tile_height)

    # Select the vector components in a subgrid
    u = meshes[0].isel(x=x_indices, y=y_indices).astype(np.float32)
    v = meshes[1].isel(x=x_indices, y=y_indices).astype(np.float32)
    # use the entire mesh for magnitude not just the sparse u,v
    mag = np.sqrt(meshes[0]**2 + meshes[1]**2)

    # Initialize a plot with appropriate axes
    fig, ax = setup_tile_plot(tile_width, tile_height)

    # If colormap background is desired, draw it now
    if colormap is not None:
        ax.imshow(
            mag,
            cmap=colormap,
            vmin=colorscale_range and colorscale_range[0],
            vmax=colorscale_range and colorscale_range[1],
            extent=(0, tile_width, 0, tile_height),
            origin="lower",
            interpolation="nearest",
        )

    # Scale the length up based on density
    u *= LENGTH_SCALE[density - 1]
    v *= LENGTH_SCALE[density - 1]
    if scaling == VectorStyleParams.GlyphScaling.CONSTANT:
        # normalize the vectors so their size is CONSTANT
        u /= mag[x_indices, y_indices]
        v /= mag[x_indices, y_indices]
    else:
        # scale up just a little
        # TODO: this should depend on dataset and its max magnitude
        u *= 3
        v *= 3

    render_args = (x_indices, y_indices, u, v)
    render_kwargs = {
        "color": color,
        "width": WIDTH[density - 1],
        "headwidth": 2.5,
        "headlength": 4,
        "headaxislength": 3.8,
        "minshaft": 1.2,
    }
    return render_vector_arrows(fig, ax, render_args, render_kwargs)


def get_colormaps() -> List[str]:
    """Returns a listing of available colormaps from matplotlib to be available for vectors."""
    return plt.colormaps()


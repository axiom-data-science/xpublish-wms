"""Visualize vector direction."""

from typing import List, Sequence
import numpy as np
from numpy.typing import NDArray
import xarray as xr

import matplotlib
from matplotlib import pyplot as plt # noqa
from PIL.Image import Image

from xpublish_wms.wms.get_map.style_types import VectorStyleParams
from xpublish_wms.wms.get_map.vectors import get_meshgrid, render_vector_arrows, setup_tile_plot


# Scale arrow length
LENGTH_SCALE = np.array([3, 2, 1]) * 9
# Other arrow size parameters are relative to the tail width
TAIL_WIDTH = np.array([3.5, 2, 1]) * 1.3
HEAD_WIDTH = [2.5, 2.5, 3.5]
# Arrow outline stroke width
LINE_WIDTH = [5, 4, 1]


def get_cell_center_indices(
    das: Sequence[xr.DataArray],
    bbox: tuple[float, float, float, float],
    width: int,
    height: int,
    density: int,
) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
    """Return (x_indices, y_indices) pixel coordinates of data cell centers within the tile.

    Uses broadcast_like to expand 1-D dimensional coords (regular grids) to the
    data's dimension order before raveling, ensuring x/y positions correspond to
    values element-wise. Both returned arrays are 1D and the same length.

    Subsampling mirrors get_meshgrid: cell centers are binned into pixel-space
    buckets of size `grid_step`; the first real cell center in each bucket is kept,
    so every arrow still anchors on an actual data cell.
    """
    x_full = das[0].x.broadcast_like(das[0])
    y_full = das[0].y.broadcast_like(das[0])
    px = ((x_full.values.ravel() - bbox[0]) / (bbox[2] - bbox[0]) * width).astype(int)
    py = ((y_full.values.ravel() - bbox[1]) / (bbox[3] - bbox[1]) * height).astype(int)

    in_tile = (px >= 0) & (px < width) & (py >= 0) & (py < height)
    px, py = px[in_tile], py[in_tile]

    # Subsample data points to prevent too many vector glyphs
    grid_step = 64 // (2 ** (density - 1))
    bucket_ids = (px // grid_step) * (height // grid_step + 1) + (py // grid_step)
    # all the buckets with more than one cell center
    _, inverse = np.unique(bucket_ids, return_inverse=True)
    # calculate a mean point for multiple cell centers by averaging their coords
    counts = np.bincount(inverse)
    px_sub = (np.bincount(inverse, weights=px) / counts).round().astype(np.intp)
    py_sub = (np.bincount(inverse, weights=py) / counts).round().astype(np.intp)

    return px_sub, py_sub


def visualize_vectors(
    meshes: Sequence[xr.DataArray],
    color: str,
    density: int,
    scaling: VectorStyleParams.GlyphScaling,
    colorscale_range: tuple[float, float] | None = None,
    colormap: str | None = None,
    draw_backing: bool = False,
    arrow_mag_color: bool = False,
    cell_center_indices: tuple[NDArray[np.intp], NDArray[np.intp]] | None = None,
) -> Image:
    """Renders a vector tile image."""
    # Create a mesh of grid-points where we will draw arrows/barbs
    if density not in (1, 2, 3):
        raise ValueError(f'Invalid density value {density}')

    # TODO during request validation make sure that vectors visualization has two layers
    assert meshes[0].shape == meshes[1].shape
    tile_height, tile_width = meshes[0].shape

    # use the entire mesh for magnitude not just the sparse u,v
    mag: xr.DataArray = np.sqrt(meshes[0]**2 + meshes[1]**2)  # type: ignore

    # Initialize a plot with appropriate axes
    fig, ax = setup_tile_plot(tile_width, tile_height)

    # If colormap background is desired, draw it now
    if draw_backing and colormap is not None:
        ax.imshow(
            mag,
            cmap=colormap,
            vmin=colorscale_range and colorscale_range[0],
            vmax=colorscale_range and colorscale_range[1],
            extent=(0, tile_width, 0, tile_height),
            origin="lower",
            interpolation="nearest",
        )

    # Create flat (1D) arrays of pixel indices where vector glyphs should be drawn.
    # This works with numpy fancy indexing
    if cell_center_indices is None:
        # A regular cartesian grid
        x_indices, y_indices = get_meshgrid(density, tile_width, tile_height)
    else:
        x_indices, y_indices = cell_center_indices
        # Filter to positions where both mesh components are finite.
        # meshes[0].values is (height, width) so index as [y, x].
        valid = (
            np.isfinite(meshes[0].values[y_indices, x_indices])
            & np.isfinite(meshes[1].values[y_indices, x_indices])
        )
        x_indices, y_indices = x_indices[valid], y_indices[valid]

    u = meshes[0].values[y_indices, x_indices].astype(np.float32)
    v = meshes[1].values[y_indices, x_indices].astype(np.float32)

    if scaling == VectorStyleParams.GlyphScaling.CONSTANT:
        # normalize the vectors so their size is CONSTANT; skip zero-magnitude
        # points to avoid inf (they'll just be drawn as zero-length arrows)
        m = mag.values[y_indices, x_indices]
        nz = m != 0
        u[nz] /= m[nz]
        v[nz] /= m[nz]
    else:
        # scale up just a little
        # TODO: this should depend on dataset and its max magnitude
        u *= 3
        v *= 3
    # Scale the length up based on density
    u *= LENGTH_SCALE[density - 1]
    v *= LENGTH_SCALE[density - 1]

    render_args = (x_indices, y_indices, u, v)
    if arrow_mag_color:
        render_args += (mag.values[y_indices, x_indices],)

    # Sum the R, G, B values and determine a contrasting edgeline
    edgecolor = "black" if sum(matplotlib.colors.to_rgb(color)) > 1.5 else "white"
    render_kwargs = {
        "color": color,
        "edgecolor": edgecolor,
        "linewidth": LINE_WIDTH[density - 1],
        "linestyle": "solid",
        "width": TAIL_WIDTH[density - 1],
        "headwidth": HEAD_WIDTH[density - 1],
        "headlength": 3,
        "headaxislength": 2.8,
        "cmap": colormap if arrow_mag_color else None,
        "vmin": colorscale_range[0] if arrow_mag_color and colorscale_range else None,
        "vmax": colorscale_range[1] if arrow_mag_color and colorscale_range else None,
    }
    return render_vector_arrows(fig, ax, render_args, render_kwargs)


def get_colormaps() -> List[str]:
    """Returns a listing of available colormaps from matplotlib to be available for vectors."""
    return plt.colormaps()


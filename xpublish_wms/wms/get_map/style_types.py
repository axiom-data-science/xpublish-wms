"""Class definitions for GetMap rendering style information."""


from typing import Literal, Sequence

from pydantic import BaseModel


class ColormapStyleParams(BaseModel):
    """Container for colormap style parameters."""
    type: Literal["colormap"]
    palettename: str
    colorscalerange: Sequence[float] | None
    autoscale: bool


class ArrowsStyleParams(BaseModel):
    """Container for arrows style parameters."""
    type: Literal["arrows"]
    color: str
    density: int


RasterStyleParams = ColormapStyleParams | ArrowsStyleParams

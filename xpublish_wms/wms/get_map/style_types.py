"""Class definitions for GetMap rendering style information."""


from enum import StrEnum
from typing import Literal, Sequence

from pydantic import BaseModel


class ColormapStyleParams(BaseModel):
    """Container for colormap style parameters."""
    type: Literal["colormap"]
    palettename: str
    colorscale_range: Sequence[float] | None
    autoscale: bool


class VectorStyleParams(BaseModel):
    """Container for vector style parameters."""
    class GlyphScaling(StrEnum):
        CONSTANT = "constant"
        UNIFORM = "uniform"
        # TODO arrow length vs overall arrow scale?

    # TODO refactor to `source = vector` and `type = arrows | barb | ...`
    type: Literal["vector"]
    color: str
    density: int
    scaling: GlyphScaling
    colorscale_range: tuple[float, float] | None
    colormap: str | None


ShadingStyleParams = ColormapStyleParams | VectorStyleParams

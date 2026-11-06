"""Static identities link to Bilt's existing physical definitions."""

from typing import ClassVar
from bilt.units import Dimension


class DimensionType:
    runtime: ClassVar[Dimension]
    absolute: ClassVar[bool] = False

"""Standalone 3x3 and 4x4 coupled linear systems for structural benchmarks."""

# ruff: noqa: INP001, D101, D103 - benchmark fixtures, not an installed package

from bilt import Implementation, Interface, equation, field, system


@system
class CoupledGroup(metaclass=Interface):
    x: float = field()
    y: float = field()
    z: float = field()
    w: float = field()
    a: float = field()
    b: float = field()
    c: float = field()
    d: float = field()


@equation
def xy(a: float, x: float, y: float) -> bool:
    return x + y == a


@equation
def yz(b: float, y: float, z: float) -> bool:
    return y + z == b


@equation
def xz(c: float, x: float, z: float) -> bool:
    return x + z == c


@equation
def zw(c: float, z: float, w: float) -> bool:
    return z + w == c


@equation
def wx(d: float, w: float, x: float) -> bool:
    return w + 2.0 * x == d


def build_group(size: int) -> tuple[Implementation, dict[str, float]]:
    """Build a square, coupled linear system with known x=1, y=2, z=3, w=4."""
    if size == 3:  # noqa: PLR2004 - benchmark has a 3-variable case
        return Implementation(CoupledGroup, [xy, yz, xz]), {"a": 3.0, "b": 5.0, "c": 4.0}
    if size == 4:  # noqa: PLR2004 - benchmark has a 4-variable case
        return Implementation(CoupledGroup, [xy, yz, zw, wx]), {"a": 3.0, "b": 5.0, "c": 7.0, "d": 6.0}
    message = f"Unsupported benchmark size: {size}"
    raise ValueError(message)

"""A Bilt Cartesian-grid subsystem, independent of GT4Py execution."""

from bilt import Implementation, Interface, dim, field, system, transform

x = dim("x")
y = dim("y")


@system
class CartesianGrid2D(metaclass=Interface):
    """Grid spacing and the inverse squared spacings used by a Laplacian."""

    dx: float = field()
    dy: float = field()
    inv_dx2: float = field()
    inv_dy2: float = field()


@transform
def inv_dx2(dx: float) -> float:
    return 1.0 / dx**2


@transform
def inv_dy2(dy: float) -> float:
    return 1.0 / dy**2


grid_implementation = Implementation(CartesianGrid2D, [inv_dx2, inv_dy2])

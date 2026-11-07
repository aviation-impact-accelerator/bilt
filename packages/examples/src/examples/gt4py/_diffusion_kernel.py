"""GT4Py stencil and Bilt field mapping for the diffusion example."""

import gt4py.next as gtx

from . import CartesianFieldMapping
from ._grid import x, y

I = gtx.Dimension("I")
J = gtx.Dimension("J")

mapping = CartesianFieldMapping(((x, I), (y, J)))


@gtx.field_operator
def laplacian_2d(
    inp: gtx.Field[[I, J], gtx.float64],
    inv_dx2: gtx.float64,
    inv_dy2: gtx.float64,
) -> gtx.Field[[I, J], gtx.float64]:
    """Five-point Laplacian on a uniform, possibly anisotropic Cartesian grid."""
    return (
        (inp(I - 1) - 2.0 * inp + inp(I + 1)) * inv_dx2
        + (inp(J - 1) - 2.0 * inp + inp(J + 1)) * inv_dy2
    )

"""GT4Py stencil and Bilt field mapping for the diffusion example."""

import gt4py.next as gtx

from bilt import dim

from . import CartesianFieldMapping

x = dim("x")
y = dim("y")
I = gtx.Dimension("I")
J = gtx.Dimension("J")

mapping = CartesianFieldMapping(((x, I), (y, J)))


@gtx.field_operator
def laplacian_2d(
    inp: gtx.Field[[I, J], gtx.float64],
) -> gtx.Field[[I, J], gtx.float64]:
    """Five-point discrete Laplacian with unit grid spacing."""
    return inp(I - 1) + inp(I + 1) + inp(J - 1) + inp(J + 1) - 4.0 * inp

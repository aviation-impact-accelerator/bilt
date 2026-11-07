"""Use a Bilt Cartesian-grid subsystem to evaluate a GT4Py stencil.

Install a compatible GT4Py Next release separately, then run:

    python -m examples.gt4py.diffusion_example
"""

import numpy as np

from bilt import Implementation, Interface, Tensor, equation, field, system, tensor, transform

from ._grid import CartesianGrid2D, grid_implementation, x, y


@system
class Diffusion(metaclass=Interface):
    temperature: Tensor = field()
    dx: float = field()
    dy: float = field()
    laplacian: Tensor = field()


@equation
def grid_dx_connection(dx: float, grid: CartesianGrid2D) -> bool:
    return dx == grid.dx


@equation
def grid_dy_connection(dy: float, grid: CartesianGrid2D) -> bool:
    return dy == grid.dy


@equation
def x_coefficient_connection(inv_dx2: float, grid: CartesianGrid2D) -> bool:
    return inv_dx2 == grid.inv_dx2


@equation
def y_coefficient_connection(inv_dy2: float, grid: CartesianGrid2D) -> bool:
    return inv_dy2 == grid.inv_dy2


@transform
def laplacian(temperature: Tensor, inv_dx2: float, inv_dy2: float) -> Tensor:
    import gt4py.next as gtx
    import numpy as np

    from examples.gt4py._diffusion_kernel import I, J, laplacian_2d, mapping

    # Bilt passes aligned NumPy arrays into compiled transforms.
    inp = mapping.to_runtime_field(temperature)

    # A periodic one-cell halo is temporary GT4Py storage, not a Bilt tensor.
    padded = np.pad(inp.asnumpy(), pad_width=1, mode="wrap")
    source = gtx.as_field([I, J], padded)
    output = gtx.as_field([I, J], np.zeros_like(padded))

    laplacian_2d(
        source,
        inv_dx2,
        inv_dy2,
        out=output[1:-1, 1:-1],
        offset_provider={},
    )
    return mapping.to_runtime_array(output[1:-1, 1:-1])


implementation = Implementation(
    Diffusion,
    [
        grid_dx_connection,
        grid_dy_connection,
        x_coefficient_connection,
        y_coefficient_connection,
        laplacian,
    ],
    internal_implementations={"grid": grid_implementation},
)


def main() -> None:
    """Check a periodic Laplacian with unequal grid spacing."""
    values = np.zeros((7, 5), dtype=np.float64)
    values[3, 2] = 1.0

    # Supply (y, x) rather than the stencil's (x, y) axis order.
    model = Diffusion(
        temperature=tensor(values.T, over=(y, x)),
        dx=2.0,
        dy=1.0,
    )

    with implementation:
        result = model.laplacian

    assert result.domain.dims == (y, x)
    assert result.shape == (5, 7)
    np.testing.assert_allclose(result.values[2, 3], -2.5)
    np.testing.assert_allclose(result.values[2, 2], 0.25)
    np.testing.assert_allclose(result.values[1, 3], 1.0)
    print("GT4Py stencil evaluated using the Bilt CartesianGrid2D subsystem")


if __name__ == "__main__":
    main()

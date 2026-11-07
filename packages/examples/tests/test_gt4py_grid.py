"""Public-API tests for the Bilt grid subsystem and its GT4Py use."""

import numpy as np
import pytest

from bilt import tensor
from examples.gt4py._grid import CartesianGrid2D, grid_implementation, x, y


def test_grid_derived_spacings() -> None:
    with grid_implementation:
        grid = CartesianGrid2D(dx=2.0, dy=4.0)
        assert grid.inv_dx2 == pytest.approx(0.25)
        assert grid.inv_dy2 == pytest.approx(0.0625)


def test_diffusion_with_grid_subsystem() -> None:
    pytest.importorskip("gt4py.next")

    from examples.gt4py.diffusion_example import Diffusion, implementation

    values = np.zeros((7, 5), dtype=np.float64)
    values[3, 2] = 1.0

    # Reverse Bilt tensor dimensions; GT4Py still computes in (I, J) order.
    model = Diffusion(
        temperature=tensor(values.T, over=(y, x)),
        dx=2.0,
        dy=1.0,
    )

    with implementation:
        result = model.laplacian

    assert result.domain.dims == (y, x)
    assert result.shape == (5, 7)
    expected = np.zeros_like(values)
    expected[3, 2] = -2.5
    expected[2, 2] = 0.25
    expected[4, 2] = 0.25
    expected[3, 1] = 1.0
    expected[3, 3] = 1.0
    np.testing.assert_allclose(result.values, expected.T)


def test_periodic_halo_reaches_opposite_boundary() -> None:
    pytest.importorskip("gt4py.next")

    from examples.gt4py.diffusion_example import Diffusion, implementation

    values = np.zeros((7, 5), dtype=np.float64)
    values[0, 0] = 1.0
    model = Diffusion(temperature=tensor(values, over=(x, y)), dx=2.0, dy=1.0)

    with implementation:
        result = model.laplacian

    assert result.domain.dims == (x, y)
    assert result.shape == (7, 5)
    assert result.values[-1, 0] == pytest.approx(0.25)
    assert result.values[0, -1] == pytest.approx(1.0)

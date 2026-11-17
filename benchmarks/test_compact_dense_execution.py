"""Benchmark end-to-end coupled tensor solving through the compact execution path.

This includes structural analysis, compilation and numerical evaluation, not just
the compact matching stage. Use `--benchmark-json` to save comparable results.
"""

import numpy as np
import pytest
from pytest_benchmark.fixture import BenchmarkFixture  # noqa: TC002 - runtime fixture resolution

from bilt._backend._jax._backend import JAX_BACKEND
from bilt._backend._python._backend import PYTHON_BACKEND
from bilt.core._structural_analysis._causalization_block_solver._compact_execution import MAX_DENSE_UNKNOWNS
from tests.core.linear.test_compact_solver_integration import _compute


@pytest.mark.parametrize(
    ("backend", "size"),
    (
        pytest.param(PYTHON_BACKEND, 4, id="python-4"),
        pytest.param(PYTHON_BACKEND, 8, id="python-8"),
        pytest.param(PYTHON_BACKEND, 16, id="python-16"),
        pytest.param(PYTHON_BACKEND, 32, id="python-32"),
        pytest.param(JAX_BACKEND, 4, id="jax-4"),
        pytest.param(JAX_BACKEND, 8, id="jax-8"),
        pytest.param(JAX_BACKEND, 16, id="jax-16"),
    ),
)
def test_compact_dense_solver_scaling(benchmark: BenchmarkFixture, backend: object, size: int) -> None:
    """Time the complete solve, checking the answer and selected execution route."""
    assert size <= MAX_DENSE_UNKNOWNS
    matrix = np.eye(size) * 3.0 + np.ones((size, size)) * (0.25 / size)
    expected = np.linspace(1.0, 2.0, size)
    rhs = matrix @ expected
    benchmark.group = "compact-dense-end-to-end"
    benchmark.extra_info["unknowns"] = size
    benchmark.extra_info["dense_matrix_entries"] = size * size
    result, intermediates = benchmark.pedantic(
        _compute,
        args=(backend, matrix, rhs),
        rounds=2,
        iterations=1,
    )
    np.testing.assert_allclose(result, expected, atol=1e-5)
    assert intermediates[0] == "__bilt_compact_linear_solution"

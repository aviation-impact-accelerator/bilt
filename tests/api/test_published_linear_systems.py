"""Published linear systems evaluated through Bilt's public modelling API.

Sources, checked 2026-10-29:
* SciPy tutorial, "Solving a linear system":
  https://docs.scipy.org/doc/scipy/tutorial/linalg.html#solving-a-linear-system
* SciPy solve examples (structured 3x3 and three positive-definite 2x2 systems):
  https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.solve.html#examples
* Intel oneMKL 2022.1, DGESV Example Program in C, input and printed Solution:
  https://www.intel.com/content/www/us/en/docs/onemkl/code-samples-lapack/2022-1/dgesv-example-c.html

Only numerical problem data are transcribed; model equations below are original.
Expected answers are published values, never another solver's runtime output.
Intel prints two decimals, so its forward check allows half a last-place unit;
a separate residual check verifies the actual solution more tightly.
"""

import typing

import numpy as np
import pytest

from bilt import (
    JAX_BACKEND,
    PYTHON_BACKEND,
    Implementation,
    Interface,
    Tensor,
    dim,
    domain,
    equation,
    field,
    system,
    tensor,
)
from bilt.core._structural_analysis._causalization_block_solver._backend import CausalizationStructuralAnalyzer
from bilt.core._system import _implementation as implementation_module

if typing.TYPE_CHECKING:
    from bilt._backend import Backend
    from bilt.symbolic._dimensions import Dim

# Tutorial publishes the exact rational result (-232, 129, 19)/25.
DENSE_A = np.array([[1.0, 3.0, 5.0], [2.0, 5.0, 1.0], [2.0, 3.0, 8.0]])
DENSE_B = np.array([10.0, 8.0, 3.0])
DENSE_X = np.array([-232.0 / 25.0, 129.0 / 25.0, 19.0 / 25.0])
STRUCTURED_A = np.array([[3.0, 2.0, 0.0], [1.0, -1.0, 0.0], [0.0, 5.0, 1.0]])
STRUCTURED_B = np.array([2.0, 4.0, -1.0])
STRUCTURED_X = np.array([2.0, -2.0, 9.0])
# Same A = a.T @ a construction and b = ones(2) as the published example.
BATCH_SEED = np.arange(12, dtype=float).reshape(3, 2, 2)
BATCH_A = BATCH_SEED.transpose(0, 2, 1) @ BATCH_SEED
BATCH_X = np.array([[1.0, -0.5], [3.0, -2.5], [5.0, -4.5]])
# Rows here follow the displayed mathematical matrices, not C column-major storage.
INTEL_A = np.array(
    [
        [6.80, -6.05, -0.45, 8.32, -9.67],
        [-2.11, -3.30, 2.58, 2.71, -5.14],
        [5.66, 5.36, -2.70, 4.35, -7.26],
        [5.97, -4.44, 0.27, -7.17, 6.08],
        [8.23, 1.08, 9.04, 2.14, -6.87],
    ]
)
INTEL_B = np.array(
    [
        [4.02, -1.56, 9.81],
        [6.19, 4.00, -4.09],
        [-8.22, -8.67, -4.57],
        [-7.57, 1.75, -8.61],
        [-3.03, 2.86, 8.99],
    ]
)
INTEL_X = np.array(
    [
        [-0.80, -0.39, 0.96],
        [-0.70, -0.55, 0.22],
        [0.59, 0.84, 1.90],
        [1.32, -0.10, 5.36],
        [0.57, 0.11, 4.04],
    ]
)
BATCH = dim("reference_case")
RHS = dim("right_hand_side")


@pytest.fixture(autouse=True)
def _use_scc(monkeypatch: pytest.MonkeyPatch) -> None:
    # Same scoped selection used by existing SCC API tests; no new public selector.
    monkeypatch.setattr(implementation_module, "CausalizationStructuralAnalyzer", CausalizationStructuralAnalyzer)


@pytest.fixture(params=(PYTHON_BACKEND, JAX_BACKEND), ids=("python", "jax"))
def backend(request: pytest.FixtureRequest) -> Backend:
    return request.param


@system
class Triple(metaclass=Interface):
    x: float = field()
    y: float = field()
    z: float = field()
    b0: float = field()
    b1: float = field()
    b2: float = field()


@equation
def dense_first(x: float, y: float, z: float, b0: float) -> bool:
    return x + 3 * y + 5 * z == b0


@equation
def dense_second(x: float, y: float, z: float, b1: float) -> bool:
    return 2 * x + 5 * y + z == b1


@equation
def dense_third(x: float, y: float, z: float, b2: float) -> bool:
    return 2 * x + 3 * y + 8 * z == b2


@equation
def dense_boundary(b0: float, inner: Triple) -> bool:
    return inner.x + 3 * inner.y + 5 * inner.z == b0


@equation
def structured_first(x: float, y: float, b0: float) -> bool:
    return 3 * x + 2 * y == b0


@equation
def structured_second(x: float, y: float, b1: float) -> bool:
    return x - y == b1


@equation
def structured_third(y: float, z: float, b2: float) -> bool:
    return 5 * y + z == b2


@equation
def structured_boundary(b0: float, inner: Triple) -> bool:
    return 3 * inner.x + 2 * inner.y == b0


def _three_model(reference: str, layout: str, *, reverse: bool = False) -> Implementation:
    first, second, third, boundary = (
        (dense_first, dense_second, dense_third, dense_boundary)
        if reference == "dense"
        else (structured_first, structured_second, structured_third, structured_boundary)
    )
    if layout == "flat":
        return Implementation(Triple, (third, second, first) if reverse else (first, second, third))
    child = Implementation(Triple, (third, second) if reverse else (second, third))
    return Implementation(Triple, [boundary], internal_implementations={"inner": child})


def _evaluate(
    model: Implementation,
    inputs: dict[str, object],
    outputs: tuple[str, ...],
    backend: Backend,
    *,
    axis: Dim | None = None,
) -> np.ndarray:
    values = []
    for output in outputs:
        # Preserve the original knowns for every output; do not turn a previously
        # computed unknown into an extra supplied constraint on the next request.
        value = model.evaluate(inputs, output, backend)
        if axis is None:
            assert not isinstance(value, Tensor)
            assert np.asarray(value).shape == ()
            values.append(np.asarray(value))
        else:
            assert isinstance(value, Tensor)
            assert value.domain == domain(axis)
            assert value.unit is None
            values.append(np.asarray(value.values))
    return np.stack(values)


def _assert_residual(a: np.ndarray, x: np.ndarray, b: np.ndarray) -> None:
    # Backward error in the original published equations, independent of any
    # rounded expected answer. Includes a small allowance for JAX float32.
    scale = np.linalg.norm(a, ord=np.inf) * np.linalg.norm(x, ord=np.inf) + np.linalg.norm(b, ord=np.inf)
    assert np.linalg.norm(a @ x - b, ord=np.inf) <= 5e-6 * scale


@pytest.mark.parametrize("reference", ("dense", "structured"))
@pytest.mark.parametrize("layout", ("flat", "parent_child"))
@pytest.mark.parametrize("reverse", (False, True), ids=("declared_order", "reverse_order"))
def test_scipy_published_three_by_three(backend: Backend, reference: str, layout: str, *, reverse: bool) -> None:
    a, b, expected = (DENSE_A, DENSE_B, DENSE_X) if reference == "dense" else (STRUCTURED_A, STRUCTURED_B, STRUCTURED_X)
    model = _three_model(reference, layout, reverse=reverse)
    inputs: dict[str, object] = {"b0": b[0]}
    prefix = "inner." if layout == "parent_child" else ""
    inputs.update({prefix + "b1": b[1], prefix + "b2": b[2]})
    actual = _evaluate(model, inputs, tuple(prefix + name for name in ("x", "y", "z")), backend)
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-6)
    _assert_residual(a, actual, b)


def test_published_dense_solution_through_fresh_instance_access(backend: Backend) -> None:
    model = _three_model("dense", "flat")
    with backend, model:
        actual = np.array([getattr(Triple(b0=10.0, b1=8.0, b2=3.0), name) for name in ("x", "y", "z")])
    np.testing.assert_allclose(actual, DENSE_X, rtol=2e-5, atol=2e-6)
    _assert_residual(DENSE_A, actual, DENSE_B)


@system
class Pair(metaclass=Interface):
    x: float = field()
    y: float = field()
    a00: float = field()
    a01: float = field()
    a11: float = field()
    b0: float = field()
    b1: float = field()


@equation
def batch_first(x: float, y: float, a00: float, a01: float, b0: float) -> bool:
    return a00 * x + a01 * y == b0


@equation
def batch_second(x: float, y: float, a01: float, a11: float, b1: float) -> bool:
    return a01 * x + a11 * y == b1


@equation
def batch_boundary(a00: float, a01: float, b0: float, inner: Pair) -> bool:
    return a00 * inner.x + a01 * inner.y == b0


@pytest.mark.parametrize("layout", ("flat", "parent_child"))
def test_scipy_published_positive_definite_batch(backend: Backend, layout: str) -> None:
    prefix = "inner." if layout == "parent_child" else ""
    model = (
        Implementation(Pair, [batch_boundary], internal_implementations={"inner": Implementation(Pair, [batch_second])})
        if prefix
        else Implementation(Pair, [batch_first, batch_second])
    )
    off_diagonal = tensor(BATCH_A[:, 0, 1], over=BATCH)
    inputs: dict[str, object] = {
        "a00": tensor(BATCH_A[:, 0, 0], over=BATCH),
        "a01": off_diagonal,
        "b0": 1.0,
        prefix + "a01": off_diagonal,
        prefix + "a11": tensor(BATCH_A[:, 1, 1], over=BATCH),
        prefix + "b1": 1.0,
    }
    actual = _evaluate(model, inputs, (prefix + "x", prefix + "y"), backend, axis=BATCH).T
    assert actual.shape == BATCH_X.shape
    # The last published A is poorly conditioned in float32; keep its exact
    # published target and allow forward error while checking backward error too.
    np.testing.assert_allclose(actual, BATCH_X, rtol=5e-4, atol=5e-5)
    for a, x in zip(BATCH_A, actual, strict=True):
        _assert_residual(a, x, np.ones(2))


@system
class Five(metaclass=Interface):
    x0: float = field()
    x1: float = field()
    x2: float = field()
    x3: float = field()
    x4: float = field()
    b0: float = field()
    b1: float = field()
    b2: float = field()
    b3: float = field()
    b4: float = field()


@equation
def intel_first(*, x0: float, x1: float, x2: float, x3: float, x4: float, b0: float) -> bool:  # noqa: PLR0913 - five unknowns and one RHS
    return 6.80 * x0 - 6.05 * x1 - 0.45 * x2 + 8.32 * x3 - 9.67 * x4 == b0


@equation
def intel_second(*, x0: float, x1: float, x2: float, x3: float, x4: float, b1: float) -> bool:  # noqa: PLR0913 - five unknowns and one RHS
    return -2.11 * x0 - 3.30 * x1 + 2.58 * x2 + 2.71 * x3 - 5.14 * x4 == b1


@equation
def intel_third(*, x0: float, x1: float, x2: float, x3: float, x4: float, b2: float) -> bool:  # noqa: PLR0913 - five unknowns and one RHS
    return 5.66 * x0 + 5.36 * x1 - 2.70 * x2 + 4.35 * x3 - 7.26 * x4 == b2


@equation
def intel_fourth(*, x0: float, x1: float, x2: float, x3: float, x4: float, b3: float) -> bool:  # noqa: PLR0913 - five unknowns and one RHS
    return 5.97 * x0 - 4.44 * x1 + 0.27 * x2 - 7.17 * x3 + 6.08 * x4 == b3


@equation
def intel_fifth(*, x0: float, x1: float, x2: float, x3: float, x4: float, b4: float) -> bool:  # noqa: PLR0913 - five unknowns and one RHS
    return 8.23 * x0 + 1.08 * x1 + 9.04 * x2 + 2.14 * x3 - 6.87 * x4 == b4


@equation
def intel_boundary(b0: float, inner: Five) -> bool:
    return 6.80 * inner.x0 - 6.05 * inner.x1 - 0.45 * inner.x2 + 8.32 * inner.x3 - 9.67 * inner.x4 == b0


@pytest.mark.parametrize("layout", ("flat", "parent_child"))
@pytest.mark.parametrize("representation", ("scalar_columns", "tensor_columns"))
def test_intel_published_dgesv_multiple_right_hand_sides(backend: Backend, layout: str, representation: str) -> None:
    prefix = "inner." if layout == "parent_child" else ""
    remaining = (intel_second, intel_third, intel_fourth, intel_fifth)
    model = (
        Implementation(Five, [intel_boundary], internal_implementations={"inner": Implementation(Five, remaining)})
        if prefix
        else Implementation(Five, (intel_first, *remaining))
    )
    outputs = tuple(prefix + f"x{i}" for i in range(5))
    if representation == "tensor_columns":
        inputs: dict[str, object] = {(prefix if i else "") + f"b{i}": tensor(INTEL_B[i], over=RHS) for i in range(5)}
        actual = _evaluate(model, inputs, outputs, backend, axis=RHS)
    else:
        columns = []
        for column in INTEL_B.T:
            inputs = {(prefix if i else "") + f"b{i}": value for i, value in enumerate(column)}
            columns.append(_evaluate(model, inputs, outputs, backend))
        actual = np.stack(columns, axis=1)
    assert actual.shape == INTEL_X.shape
    # Match Intel's two-decimal printed answer, not invented extra digits.
    np.testing.assert_allclose(actual, INTEL_X, rtol=0.0, atol=0.00501)
    _assert_residual(INTEL_A, actual, INTEL_B)

import typing

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bilt import Domain, Implementation, Interface, Quantity, Tensor, dim, domain, equation, field, system, tensor
from bilt._backend._errors import BiltUnsupportedNodeError
from bilt._backend._jax._backend import JAX_BACKEND
from bilt._backend._python._backend import PYTHON_BACKEND
from bilt.core._evaluation_context import EvaluationContext
from bilt.core._evaluation_request import EvaluationRequest
from bilt.core._structural_analysis._causalization_block_solver._backend import CausalizationStructuralAnalyzer
from bilt.core._structural_analysis._causalization_block_solver._errors import BiltExecutionPlanError
from bilt.core._system import _implementation as implementation_module
from bilt.errors import BiltLinearSolveError
from bilt.symbolic._dimensions import coordinate_axis
from bilt.units.dimensions import LENGTH
from bilt.units.units import kilometer, meter

if typing.TYPE_CHECKING:
    import ast

    from bilt._backend import Backend, BackendCallable

TIME = dim("time")
SITE = dim("site")
OverTime = typing.Annotated[Tensor, domain(TIME)]
Shared = typing.Annotated[Tensor, Domain.scalar()]
LengthArray = typing.Annotated[Tensor, LENGTH]
LengthScalar = typing.Annotated[Quantity, LENGTH]


@pytest.fixture(autouse=True)
def _use_scc(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(implementation_module, "ExhaustiveStructuralAnalyser", CausalizationStructuralAnalyzer)


@pytest.fixture(params=(PYTHON_BACKEND, JAX_BACKEND), ids=("python", "jax"))
def backend(request: pytest.FixtureRequest) -> Backend:
    return request.param


@system
class Ports(metaclass=Interface):
    x: Tensor = field()
    y: Tensor = field()
    p: Tensor = field()
    q: float | Tensor = field()
    a: float | Tensor = field()
    answer: float | Tensor = field()


@equation
def total(x: Tensor, y: Tensor, p: Tensor, a: Tensor) -> bool:
    return a * x + y == p


@equation
def feedback(inner: Ports, q: Tensor) -> bool:
    return inner.x - inner.y == q


def _model() -> Implementation:
    return Implementation(Ports, [feedback], internal_implementations={"inner": Implementation(Ports, [total])})


@pytest.mark.parametrize("output", ("inner.x", "inner.y"))
@pytest.mark.parametrize("array_coefficient", (False, True))
def test_cross_boundary_unknowns_share_the_broadcast_domain(
    backend: Backend, output: str, *, array_coefficient: bool
) -> None:
    coefficients = np.array([1.0, 2.0, 3.0]) if array_coefficient else np.ones(3)
    result = _model().evaluate(
        {
            "inner.p": tensor([10.0, 20.0, 30.0], over=TIME),
            "inner.a": tensor(coefficients, over=TIME) if array_coefficient else 1.0,
            "q": 2.0,
        },
        output,
        backend,
    )
    expected = (np.array([10.0, 20.0, 30.0]) + 2) / (coefficients + 1)
    if output == "inner.y":
        expected -= 2
    assert isinstance(result, Tensor)
    assert result.domain == domain(TIME)
    np.testing.assert_allclose(result.values, expected, rtol=1e-5)


@pytest.mark.parametrize("reverse_input", (False, True))
@pytest.mark.parametrize("reverse_context", (False, True))
def test_independent_axes_and_input_axis_order_follow_the_canonical_domain(
    backend: Backend, *, reverse_input: bool, reverse_context: bool
) -> None:
    p = np.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]])
    axes = (TIME, SITE) if reverse_context else (SITE, TIME)
    with EvaluationContext() as evaluation:
        for axis in axes:
            coordinates = ("t0", "t1", "t2") if axis == TIME else ("s0", "s1")
            evaluation.domains.bind(coordinate_axis(axis, coordinates))
        result = _model().evaluate(
            {
                "inner.p": tensor(p.T if reverse_input else p, over=(TIME, SITE) if reverse_input else (SITE, TIME)),
                "inner.a": tensor([1.0, 2.0, 3.0], over=TIME),
                "q": tensor([2.0, 4.0], over=SITE),
            },
            "inner.x",
            backend,
        )
    expected = (p + np.array([2.0, 4.0])[:, None]) / np.array([2.0, 3.0, 4.0])[None, :]
    assert isinstance(result, Tensor)
    assert result.domain == domain(*axes)
    np.testing.assert_allclose(result.values, expected.T if reverse_context else expected, rtol=1e-5)


def test_tensor_coefficients_can_supply_axes_with_scalar_rhs(backend: Backend) -> None:
    result = _model().evaluate(
        {"inner.p": 10.0, "inner.a": tensor([1.0, 2.0], over=TIME), "q": 2.0}, "inner.x", backend
    )
    assert isinstance(result, Tensor)
    assert result.domain == domain(TIME)
    np.testing.assert_allclose(result.values, [6.0, 4.0], rtol=1e-5)


def test_one_batched_matrix_is_assembled_per_coupled_block(monkeypatch: pytest.MonkeyPatch) -> None:
    original = np.linalg.solve
    shapes = []

    def solve(matrix: np.ndarray, rhs: np.ndarray) -> np.ndarray:
        shapes.append(matrix.shape)
        return original(matrix, rhs)

    monkeypatch.setattr(np.linalg, "solve", solve)
    result = _model().evaluate(
        {"inner.p": tensor([10.0, 20.0, 30.0], over=TIME), "inner.a": 1.0, "q": 2.0}, "inner.x", PYTHON_BACKEND
    )
    assert isinstance(result, Tensor)
    np.testing.assert_allclose(result.values, [6.0, 11.0, 16.0])
    assert shapes == [(3, 2, 2)]


@pytest.mark.parametrize("output", ("inner.x", "inner.y"))
def test_unitful_values_are_lowered_and_lifted_across_the_boundary(backend: Backend, output: str) -> None:
    @system
    class Lengths(metaclass=Interface):
        x: LengthArray = field()
        y: LengthArray = field()
        p: LengthArray = field()
        q: LengthScalar = field()

    @equation
    def balance(x: LengthArray, y: LengthArray, p: LengthArray) -> bool:
        return x + y == p

    @equation
    def relation(inner: Lengths, q: LengthScalar) -> bool:
        return inner.x - inner.y == q

    parent = Implementation(Lengths, [relation], internal_implementations={"inner": Implementation(Lengths, [balance])})
    result = parent.evaluate(
        {"inner.p": tensor([0.010, 0.020], over=TIME, unit=kilometer), "q": 2.0 * meter}, output, backend
    )
    assert isinstance(result, Tensor)
    assert result.unit == meter
    assert result.domain == domain(TIME)
    np.testing.assert_allclose(result.values, [6.0, 11.0] if output == "inner.x" else [4.0, 9.0], rtol=1e-5)


@pytest.mark.parametrize("size", (0, 1))
def test_empty_and_singleton_batches_preserve_the_named_axis(backend: Backend, size: int) -> None:
    values = np.arange(size, dtype=float) + 10
    result = _model().evaluate({"inner.p": tensor(values, over=TIME), "inner.a": 1.0, "q": 2.0}, "inner.x", backend)
    assert isinstance(result, Tensor)
    assert result.domain == domain(TIME)
    assert result.values.shape == (size,)
    np.testing.assert_allclose(result.values, (values + 2) / 2, rtol=1e-5)


def test_cached_compilation_accepts_changing_extents_without_recompiling(monkeypatch: pytest.MonkeyPatch) -> None:
    original = PYTHON_BACKEND.transpile_linear_solve
    calls = []

    def compile_solve(function: ast.FunctionDef) -> BackendCallable:
        calls.append(function)
        return original(function)

    monkeypatch.setattr(PYTHON_BACKEND, "transpile_linear_solve", compile_solve)
    parent = _model()
    for values in ([10.0, 20.0, 30.0], [4.0, 8.0]):
        result = parent.evaluate(
            {"inner.p": tensor(values, over=TIME), "inner.a": 1.0, "q": 2.0}, "inner.x", PYTHON_BACKEND
        )
        assert isinstance(result, Tensor)
        np.testing.assert_allclose(result.values, (np.array(values) + 2) / 2)
    assert len(calls) == 1


@pytest.mark.parametrize("depth", (1, 2))
def test_broadcast_block_can_span_multiple_nested_boundaries(backend: Backend, depth: int) -> None:
    @equation
    def supply(p: Tensor, inner: Ports) -> bool:
        return inner.p == p

    @equation
    def coefficient(a: Tensor, inner: Ports) -> bool:
        return inner.a == a

    @equation
    def expose(x: Tensor, inner: Ports) -> bool:
        return x == inner.x

    parent = Implementation(
        Ports,
        [supply, coefficient, expose, feedback],
        internal_implementations={"inner": Implementation(Ports, [total])},
    )
    for _ in range(depth):

        @equation
        def boundary_q(q: Tensor, inner: Ports) -> bool:
            return inner.q == q

        parent = Implementation(
            Ports, [supply, coefficient, expose, boundary_q], internal_implementations={"inner": parent}
        )
    result = parent.evaluate({"p": tensor([10.0, 20.0], over=TIME), "a": 1.0, "q": 2.0}, "x", backend)
    assert isinstance(result, Tensor)
    assert result.domain == domain(TIME)
    np.testing.assert_allclose(result.values, [6.0, 11.0], rtol=1e-5)


def test_broadcast_output_domain_reaches_a_downstream_reduction(backend: Backend) -> None:
    @equation
    def expose(x: Tensor, inner: Ports) -> bool:
        return x == inner.x

    @equation
    def reduction(answer: float, x: Tensor) -> bool:
        return answer == x.sum(TIME)

    parent = Implementation(
        Ports, [feedback, expose, reduction], internal_implementations={"inner": Implementation(Ports, [total])}
    )
    result = parent.evaluate({"inner.p": tensor([10.0, 20.0], over=TIME), "inner.a": 1.0, "q": 2.0}, "answer", backend)
    assert not isinstance(result, Tensor)
    assert np.asarray(result).shape == ()
    assert result == pytest.approx(17.0)


def test_explicit_common_unknown_domains_are_preserved(backend: Backend) -> None:
    @equation
    def hinted(x: OverTime, y: OverTime, p: OverTime) -> bool:
        return x + y == p

    parent = Implementation(Ports, [feedback], internal_implementations={"inner": Implementation(Ports, [hinted])})
    result = parent.evaluate({"inner.p": tensor([10.0, 20.0], over=TIME), "q": 2.0}, "inner.y", backend)
    assert isinstance(result, Tensor)
    assert result.domain == domain(TIME)
    np.testing.assert_allclose(result.values, [4.0, 9.0], rtol=1e-5)


def test_incompatible_shared_and_tensor_equation_layouts_are_rejected(backend: Backend) -> None:
    @equation
    def hinted(x: OverTime, y: Shared, p: Tensor) -> bool:
        return x + y == p

    parent = Implementation(Ports, [feedback], internal_implementations={"inner": Implementation(Ports, [hinted])})
    with pytest.raises(BiltExecutionPlanError, match="same domain layout"):
        parent.evaluate({"inner.p": tensor([10.0, 20.0], over=TIME), "q": 2.0}, "inner.x", backend)


def test_hinted_input_must_match_its_qualified_domain(backend: Backend) -> None:
    @equation
    def hinted(x: Tensor, y: Tensor, p: OverTime) -> bool:
        return x + y == p

    parent = Implementation(Ports, [feedback], internal_implementations={"inner": Implementation(Ports, [hinted])})
    with pytest.raises(BiltExecutionPlanError, match=r"inner.p.*Domain hint"):
        parent.evaluate({"inner.p": tensor([10.0, 20.0], over=SITE), "q": 2.0}, "inner.x", backend)


def test_joint_cross_boundary_reductions_require_explicit_hints(backend: Backend) -> None:
    @equation
    def reduction(x: Tensor, y: Tensor, p: Tensor) -> bool:
        return x.sum(TIME) + y == p

    parent = Implementation(Ports, [feedback], internal_implementations={"inner": Implementation(Ports, [reduction])})
    with pytest.raises(BiltExecutionPlanError, match="explicit Domain hints"):
        parent.evaluate({"inner.p": tensor([10.0, 20.0], over=TIME), "q": 2.0}, "inner.x", backend)


def test_singular_coordinate_does_not_poison_later_values(backend: Backend) -> None:
    parent = _model()
    values: dict[str, object] = {"inner.p": tensor([10.0, 20.0], over=TIME), "q": 2.0}
    error = jax.errors.JaxRuntimeError if backend is JAX_BACKEND else BiltLinearSolveError
    with pytest.raises(error, match="singular"):
        parent.evaluate(values | {"inner.a": tensor([-1.0, 1.0], over=TIME)}, "inner.x", backend)
    result = parent.evaluate(values | {"inner.a": tensor([1.0, 1.0], over=TIME)}, "inner.x", backend)
    assert isinstance(result, Tensor)
    np.testing.assert_allclose(result.values, [6.0, 11.0], rtol=1e-5)


def test_complex_coefficients_remain_complex(backend: Backend) -> None:
    result = _model().evaluate({"inner.p": 10.0, "inner.a": tensor([1j, 2j], over=TIME), "q": 2.0}, "inner.x", backend)
    assert isinstance(result, Tensor)
    np.testing.assert_allclose(result.values, 12 / (np.array([1j, 2j]) + 1), rtol=1e-5)


def test_compile_rejection_retries_a_broadcast_formulation(monkeypatch: pytest.MonkeyPatch) -> None:
    original = PYTHON_BACKEND.transpile_linear_solve
    calls = []

    def compile_solve(function: ast.FunctionDef) -> BackendCallable:
        calls.append(function)
        if len(calls) == 1:
            message = "broadcast formulation refused"
            raise BiltUnsupportedNodeError(message)
        return original(function)

    monkeypatch.setattr(PYTHON_BACKEND, "transpile_linear_solve", compile_solve)
    result = _model().evaluate(
        {"inner.p": tensor([10.0, 20.0], over=TIME), "inner.a": 1.0, "q": 2.0}, "inner.x", PYTHON_BACKEND
    )
    assert isinstance(result, Tensor)
    np.testing.assert_allclose(result.values, [6.0, 11.0])
    assert len(calls) == len((total, feedback))


def test_jax_broadcast_cross_boundary_solve_remains_differentiable() -> None:
    parent = _model()
    with EvaluationContext() as evaluation:
        evaluation.domains.bind(tensor([10.0, 20.0], over=TIME))
        function = parent.get_evaluation(
            EvaluationRequest(
                output="inner.x",
                input_domains={"inner.p": domain(TIME), "inner.a": Domain.scalar(), "q": Domain.scalar()},
            ),
            JAX_BACKEND,
        )
    derivative = jax.jacrev(lambda p: function.callable(inner__p=p, inner__a=1.0, q=2.0))(jnp.array([10.0, 20.0]))
    np.testing.assert_allclose(derivative, 0.5 * np.eye(2), rtol=1e-5)


def test_unrelated_named_axes_are_not_added_to_the_output_domain(backend: Backend) -> None:
    result = _model().evaluate(
        {
            "inner.p": tensor([10.0, 20.0], over=TIME),
            "inner.a": 1.0,
            "q": 2.0,
            "answer": tensor([1.0, 2.0, 3.0], over=SITE),
        },
        "inner.x",
        backend,
    )
    assert isinstance(result, Tensor)
    assert result.domain == domain(TIME)
    np.testing.assert_allclose(result.values, [6.0, 11.0], rtol=1e-5)


def test_coordinate_sensitive_downstream_selection_uses_the_current_context(backend: Backend) -> None:
    @equation
    def expose(x: Tensor, inner: Ports) -> bool:
        return x == inner.x

    @equation
    def selected(answer: float, x: Tensor) -> bool:
        return answer == x.sel(TIME, "reference")

    parent = Implementation(
        Ports, [feedback, expose, selected], internal_implementations={"inner": Implementation(Ports, [total])}
    )
    for coordinates, values, expected in (
        (("reference", "other"), [10.0, 20.0], 6.0),
        (("other", "reference"), [10.0, 20.0], 11.0),
        (("other", "third", "reference"), [10.0, 20.0, 30.0], 16.0),
    ):
        with EvaluationContext() as evaluation:
            evaluation.domains.bind(coordinate_axis(TIME, coordinates))
            result = parent.evaluate(
                {"inner.p": tensor(values, over=TIME), "inner.a": 1.0, "q": 2.0}, "answer", backend
            )
        assert np.asarray(result).shape == ()
        assert result == pytest.approx(expected, rel=1e-5)


def test_two_coupled_regions_keep_separate_batched_matrices(backend: Backend, monkeypatch: pytest.MonkeyPatch) -> None:
    original = np.linalg.solve
    shapes = []

    def solve(matrix: np.ndarray, rhs: np.ndarray) -> np.ndarray:
        shapes.append(matrix.shape)
        return original(matrix, rhs)

    monkeypatch.setattr(np.linalg, "solve", solve)

    @equation
    def next_total(x: Tensor, y: Tensor, inner: Ports) -> bool:
        return x + y == inner.x

    @equation
    def next_difference(x: Tensor, y: Tensor, inner: Ports) -> bool:
        return x - y == inner.y

    parent = Implementation(
        Ports,
        [feedback, next_total, next_difference],
        internal_implementations={"inner": Implementation(Ports, [total])},
    )
    result = parent.evaluate({"inner.p": tensor([10.0, 20.0], over=TIME), "inner.a": 1.0, "q": 2.0}, "x", backend)
    assert isinstance(result, Tensor)
    assert result.domain == domain(TIME)
    np.testing.assert_allclose(result.values, [5.0, 10.0], rtol=1e-5)
    if backend is PYTHON_BACKEND:
        assert shapes == [(2, 2, 2), (2, 2, 2)]


def test_rejected_tensor_layout_does_not_reject_a_later_scalar_request(backend: Backend) -> None:
    @equation
    def hinted(x: Shared, y: Shared, p: Tensor) -> bool:
        return x + y == p

    parent = Implementation(Ports, [feedback], internal_implementations={"inner": Implementation(Ports, [hinted])})
    with pytest.raises(BiltExecutionPlanError, match="same domain layout"):
        parent.evaluate({"inner.p": tensor([10.0, 20.0], over=TIME), "q": 2.0}, "inner.x", backend)
    assert parent.evaluate({"inner.p": 10.0, "q": 2.0}, "inner.x", backend) == pytest.approx(6.0)


def test_sibling_instances_can_share_code_and_have_distinct_broadcast_values(backend: Backend) -> None:
    @equation
    def first(left: Ports, right: Ports, q: Tensor) -> bool:
        return left.y - right.x == q

    @equation
    def second(left: Ports, right: Ports) -> bool:
        return right.y == 2 * left.x

    child = Implementation(Ports, [total])
    parent = Implementation(Ports, [first, second], internal_implementations={"left": child, "right": child})
    result = parent.evaluate(
        {
            "left.p": tensor([10.0, 20.0], over=TIME),
            "right.p": tensor([14.0, 28.0], over=TIME),
            "left.a": 1.0,
            "right.a": 1.0,
            "q": 2.0,
        },
        "left.x",
        backend,
    )
    assert isinstance(result, Tensor)
    assert result.domain == domain(TIME)
    np.testing.assert_allclose(result.values, [6.0, 10.0], rtol=1e-5)

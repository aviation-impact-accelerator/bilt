import typing

import numpy as np
import pytest

from bilt import Implementation, Interface, Quantity, Tensor, dim, equation, field, system, tensor, transform
from bilt._backend._errors import BiltUnsupportedNodeError
from bilt._backend._jax._backend import JAX_BACKEND
from bilt._backend._python._backend import PYTHON_BACKEND
from bilt.core._evaluation_context import EvaluationContext
from bilt.core._structural_analysis._causalization_block_solver import _search as search
from bilt.core._structural_analysis._causalization_block_solver._backend import CausalizationStructuralAnalyzer
from bilt.core._structural_analysis._causalization_block_solver._errors import BiltExecutionPlanError
from bilt.core._sub_system import SubSystemConstituent
from bilt.core._system import _implementation as implementation_module
from bilt.symbolic._dimensions import coordinate_axis
from bilt.symbolic._domains import domain
from bilt.units.dimensions import LENGTH
from bilt.units.units import kilometer, meter

if typing.TYPE_CHECKING:
    import ast

    from bilt._backend import Backend, BackendCallable

TIME = dim("time")
LengthArray = typing.Annotated[Tensor, LENGTH]
LengthScalar = typing.Annotated[Quantity, LENGTH]


@pytest.fixture(autouse=True)
def _use_scc(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(implementation_module, "ExhaustiveStructuralAnalyser", CausalizationStructuralAnalyzer)


@pytest.fixture(params=(PYTHON_BACKEND, JAX_BACKEND), ids=("python", "jax"))
def backend(request: pytest.FixtureRequest) -> Backend:
    return request.param


@system
class Pair(metaclass=Interface):
    x: float | Tensor = field()
    y: float | Tensor = field()
    p: float | Tensor = field()
    q: float | Tensor = field()


@equation
def balance(x: Tensor, y: Tensor, p: Tensor) -> bool:
    return x + y == p


@equation
def difference(x: Tensor, y: Tensor, q: Tensor) -> bool:
    return x - y == q


@equation
def aggregate(x: Tensor, y: Tensor, q: float) -> bool:
    return 0.5 * x.sum(TIME) - y == q


def _parent(child: Implementation) -> Implementation:
    @equation
    def supply_p(p: Tensor, inner: Pair) -> bool:
        return inner.p == 2 * p

    @equation
    def supply_q(q: Tensor, inner: Pair) -> bool:
        return inner.q == q + 1

    @equation
    def expose_x(x: Tensor, inner: Pair) -> bool:
        return x == inner.x

    @equation
    def expose_y(y: Tensor, inner: Pair) -> bool:
        return y == inner.y

    return Implementation(Pair, [supply_p, supply_q, expose_x, expose_y], internal_implementations={"inner": child})


@pytest.mark.parametrize("depth", (1, 2))
@pytest.mark.parametrize("output", ("x", "y"))
@pytest.mark.parametrize("tensor_inputs", (False, True))
def test_child_solves_pointwise_block_after_parent_computes_inputs(
    backend: Backend, depth: int, output: str, *, tensor_inputs: bool
) -> None:
    implementation = Implementation(Pair, [balance, difference])
    for _ in range(depth):
        implementation = _parent(implementation)
    values: dict[str, object] = {"p": tensor([2.0, 4.0], over=TIME) if tensor_inputs else 2.0, "q": 1.0}
    result = implementation.evaluate(values, output, backend)
    p = np.array([2.0, 4.0]) if tensor_inputs else np.array(2.0)
    expected = (2**depth * p + (1 + depth) * (1 if output == "x" else -1)) / 2
    if tensor_inputs:
        assert isinstance(result, Tensor)
        assert result.domain == domain(TIME)
        np.testing.assert_allclose(result.values, expected, atol=1e-5)
    else:
        np.testing.assert_allclose(np.asarray(result), expected, atol=1e-5)


@pytest.mark.parametrize("depth", (1, 2))
@pytest.mark.parametrize("output", ("x", "y"))
def test_child_infers_shared_unknown_and_reduction_domains(backend: Backend, depth: int, output: str) -> None:
    implementation = Implementation(Pair, [balance, aggregate])
    for _ in range(depth):
        implementation = _parent(implementation)
    p = np.array([2.0, 4.0, 6.0]) * 2**depth
    shared = (0.5 * p.sum() - (1 + depth)) / 2.5
    result = implementation.evaluate({"p": tensor([2.0, 4.0, 6.0], over=TIME), "q": 1.0}, output, backend)
    if output == "x":
        assert isinstance(result, Tensor)
        assert result.domain == domain(TIME)
        np.testing.assert_allclose(result.values, p - shared, atol=1e-5)
    else:
        assert not isinstance(result, Tensor)
        np.testing.assert_allclose(np.asarray(result), shared, atol=1e-5)


def test_child_route_query_exposes_only_required_boundary_inputs(backend: Backend) -> None:
    child = SubSystemConstituent("inner", Implementation(Pair, [balance, aggregate]))
    assert child.required_inputs_for("inner.x", backend, within={"inner.p", "inner.q"}) == frozenset({"p", "q"})


@pytest.mark.parametrize("output", ("x", "y"))
def test_child_units_survive_shared_and_tensor_output_lifting(backend: Backend, output: str) -> None:
    @system
    class Lengths(metaclass=Interface):
        x: LengthArray = field()
        y: LengthScalar = field()
        p: LengthArray = field()
        q: LengthScalar = field()

    @equation
    def total(x: LengthArray, y: LengthScalar, p: LengthArray) -> bool:
        return x + y == p

    @equation
    def reduction(x: LengthArray, y: LengthScalar, q: LengthScalar) -> bool:
        return 0.5 * x.sum(TIME) - y == q

    child = Implementation(Lengths, [total, reduction])
    parent = Implementation(Pair, [SubSystemConstituent("inner", child)])
    result = parent.evaluate(
        {"inner.p": tensor([0.002, 0.004, 0.006], over=TIME, unit=kilometer), "inner.q": 1.0 * meter},
        "inner." + output,
        backend,
    )
    if output == "x":
        assert isinstance(result, Tensor)
        assert result.unit == meter
        np.testing.assert_allclose(result.values, [0, 2, 4], atol=1e-5)
    else:
        assert isinstance(result, Quantity)
        assert result.unit == meter
        assert float(result.value) == pytest.approx(2.0, abs=1e-5)


def test_child_selection_cache_respects_coordinates_and_extents(backend: Backend) -> None:
    @equation
    def selected(x: Tensor, y: Tensor, q: float) -> bool:
        return x.sel(TIME, "reference") - y == q

    parent = _parent(Implementation(Pair, [balance, selected]))
    for coordinates, values, expected in (
        (("reference", "other"), [2.0, 4.0], 1.0),
        (("other", "reference"), [2.0, 4.0], 3.0),
        (("other", "third", "reference"), [2.0, 4.0, 6.0], 5.0),
    ):
        with EvaluationContext() as evaluation:
            evaluation.domains.bind(coordinate_axis(TIME, coordinates))
            result = parent.evaluate({"p": tensor(values, over=TIME), "q": 1.0}, "y", backend)
        assert result == pytest.approx(expected, abs=1e-5)


def test_child_retries_another_block_formulation_after_compilation_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    original = PYTHON_BACKEND.transpile_linear_solve
    attempts = []

    def compile_solve(function: ast.FunctionDef) -> BackendCallable:
        attempts.append(function)
        if len(attempts) == 1:
            message = "child block formulation refused"
            raise BiltUnsupportedNodeError(message)
        return original(function)

    monkeypatch.setattr(PYTHON_BACKEND, "transpile_linear_solve", compile_solve)
    parent = _parent(Implementation(Pair, [balance, difference]))
    result = parent.evaluate({"p": 2.0, "q": 1.0}, "x", PYTHON_BACKEND)
    assert result == pytest.approx(3.0)
    assert len(attempts) == len((balance, difference))


def test_child_rejections_terminate_and_a_new_request_can_compile(monkeypatch: pytest.MonkeyPatch) -> None:
    parent = _parent(Implementation(Pair, [balance, difference]))

    def refuse(_: ast.FunctionDef) -> typing.Never:
        message = "all child blocks refused"
        raise BiltUnsupportedNodeError(message)

    with monkeypatch.context() as patch:
        patch.setattr(PYTHON_BACKEND, "transpile_linear_solve", refuse)
        with pytest.raises(BiltExecutionPlanError, match="all child blocks refused"):
            parent.evaluate({"p": 2.0, "q": 1.0}, "x", PYTHON_BACKEND)
    assert parent.evaluate({"p": 2.0, "q": 1.0}, "x", PYTHON_BACKEND) == pytest.approx(3.0)


def _unexpected_search(*_: object, **__: object) -> typing.Never:
    message = "Unsupported unavoidable coupling must be rejected before execution search."
    raise AssertionError(message)


def test_coupling_across_parent_and_child_is_still_rejected_early(monkeypatch: pytest.MonkeyPatch) -> None:
    difference_value = 2.0

    @equation
    def feedback(inner: Pair) -> bool:
        return inner.x - inner.y == difference_value

    parent = Implementation(Pair, [feedback], internal_implementations={"inner": Implementation(Pair, [balance])})
    monkeypatch.setattr(search, "find_matching", _unexpected_search)
    with pytest.raises(BiltExecutionPlanError, match=r"Required coupled computations: inner\.x, inner\.y"):
        parent.evaluate({"inner.p": 4.0}, "inner.x", PYTHON_BACKEND)


@pytest.mark.parametrize("directional", (False, True))
def test_unsupported_child_coupling_is_still_rejected_early(
    monkeypatch: pytest.MonkeyPatch, *, directional: bool
) -> None:
    @equation
    def nonlinear(x: Tensor, y: Tensor, q: Tensor) -> bool:
        return x * y == q

    @transform
    def x(y: float) -> float:
        return y + 1

    @transform
    def y(x: float) -> float:
        return x + 1

    child = Implementation(Pair, [x, y] if directional else [balance, nonlinear])
    parent = Implementation(Pair, [SubSystemConstituent("inner", child)])
    monkeypatch.setattr(search, "find_matching", _unexpected_search)
    values: dict[str, object] = {} if directional else {"inner.p": 4.0, "inner.q": 2.0}
    with pytest.raises(BiltExecutionPlanError, match=r"Required coupled computations: inner\.x, inner\.y"):
        parent.evaluate(values, "inner.x", PYTHON_BACKEND)


def test_sibling_instances_share_code_but_not_block_values(backend: Backend) -> None:
    @equation
    def combine(x: Tensor, left: Pair, right: Pair) -> bool:
        return x == left.x + right.x

    child = Implementation(Pair, [balance, difference])
    parent = Implementation(Pair, [combine], internal_implementations={"left": child, "right": child})
    result = parent.evaluate({"left.p": 4.0, "left.q": 2.0, "right.p": 10.0, "right.q": 2.0}, "x", backend)
    assert result == pytest.approx(9.0)


def test_coupling_between_instances_of_one_implementation_is_still_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    @equation
    def left_feedback(left: Pair, right: Pair, q: float) -> bool:
        return left.x - 2 * right.y == q

    @equation
    def right_feedback(left: Pair, right: Pair, q: float) -> bool:
        return right.x - 2 * left.y == q

    child = Implementation(Pair, [balance])
    parent = Implementation(
        Pair, [left_feedback, right_feedback], internal_implementations={"left": child, "right": child}
    )
    monkeypatch.setattr(search, "find_matching", _unexpected_search)
    with pytest.raises(BiltExecutionPlanError, match="Required coupled computations"):
        parent.evaluate({"left.p": 4.0, "right.p": 8.0, "q": 1.0}, "left.x", PYTHON_BACKEND)

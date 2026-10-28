import typing

import jax
import numpy as np
import pytest

from bilt import Implementation, Interface, Quantity, Tensor, dim, equation, field, system, tensor, transform
from bilt._backend._errors import BiltUnsupportedNodeError
from bilt._backend._jax._backend import JAX_BACKEND
from bilt._backend._python._backend import PYTHON_BACKEND
from bilt.core._structural_analysis._causalization_block_solver._backend import CausalizationStructuralAnalyzer
from bilt.core._structural_analysis._causalization_block_solver._errors import BiltExecutionPlanError
from bilt.core._sub_system import SubSystemConstituent
from bilt.core._system import _implementation as implementation_module
from bilt.errors import BiltLinearSolveError
from bilt.units.dimensions import LENGTH
from bilt.units.units import kilometer, meter

if typing.TYPE_CHECKING:
    import ast

    from bilt._backend import Backend, BackendCallable

Length = typing.Annotated[Quantity, LENGTH]


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
    raw: float = field()
    answer: float = field()


@equation
def balance(x: float, y: float, p: float) -> bool:
    return x + y == p


@equation
def feedback(inner: Pair, q: float) -> bool:
    return inner.x - inner.y == q


def _crossing(*, reverse: bool = False) -> Implementation:
    child = SubSystemConstituent("inner", Implementation(Pair, [balance]))
    operations = [child, feedback]
    return Implementation(Pair, operations[::-1] if reverse else operations)


@pytest.mark.parametrize(("output", "expected"), (("inner.x", 6.0), ("inner.y", 4.0)))
@pytest.mark.parametrize("reverse", (False, True))
def test_parent_feedback_and_child_balance_form_one_scalar_solve(
    backend: Backend, output: str, expected: float, *, reverse: bool
) -> None:
    result = _crossing(reverse=reverse).evaluate({"inner.p": 10.0, "q": 2.0}, output, backend)
    assert result == pytest.approx(expected)


def test_parent_transform_can_supply_a_block_input_and_consume_its_result(backend: Backend) -> None:
    @transform
    def p(raw: float) -> float:
        return raw * 2

    @equation
    def supply(p: float, inner: Pair) -> bool:
        return inner.p == p

    @equation
    def expose(x: float, inner: Pair) -> bool:
        return x == inner.x

    @transform
    def answer(x: float) -> float:
        return x**2 + 1

    parent = Implementation(
        Pair, [p, supply, expose, answer, feedback], internal_implementations={"inner": Implementation(Pair, [balance])}
    )
    assert parent.evaluate({"raw": 5.0, "q": 2.0}, "answer", backend) == pytest.approx(37.0)


def _nested_crossing(depth: int) -> Implementation:
    @equation
    def supply(p: float, inner: Pair) -> bool:
        return inner.p == p

    @equation
    def expose(x: float, inner: Pair) -> bool:
        return x == inner.x

    implementation = Implementation(
        Pair, [supply, expose, feedback], internal_implementations={"inner": Implementation(Pair, [balance])}
    )
    for _ in range(depth):

        @equation
        def supply_q(q: float, inner: Pair) -> bool:
            return inner.q == q

        implementation = Implementation(
            Pair, [supply, supply_q, expose], internal_implementations={"inner": implementation}
        )
    return implementation


@pytest.mark.parametrize("depth", (0, 1, 2))
def test_cross_boundary_block_inside_nested_children(backend: Backend, depth: int) -> None:
    implementation = _nested_crossing(depth)
    assert implementation.evaluate({"p": 10.0, "q": 2.0}, "x", backend) == pytest.approx(6.0)


def test_required_input_query_reaches_a_cross_boundary_child_block(backend: Backend) -> None:
    child = SubSystemConstituent("module", _nested_crossing(0))
    assert child.required_inputs_for("module.x", backend, within={"module.p", "module.q"}) == frozenset({"p", "q"})


def test_parent_and_child_unit_declarations_and_quantity_constants_survive_the_joint_solve(backend: Backend) -> None:
    @system
    class Lengths(metaclass=Interface):
        x: Length = field()
        y: Length = field()
        p: Length = field()

    @system
    class Parent(metaclass=Interface):
        q: Length = field()

    @equation
    def total(x: Length, y: Length, p: Length) -> bool:
        return x + y == p + 2.0 * meter

    @equation
    def relation(inner: Lengths, q: Length) -> bool:
        return inner.x - inner.y == q

    parent = Implementation(Parent, [relation], internal_implementations={"inner": Implementation(Lengths, [total])})
    result = parent.evaluate({"inner.p": 0.010 * kilometer, "q": 2.0 * meter}, "inner.x", backend)
    assert isinstance(result, Quantity)
    assert result.unit == meter
    assert float(result.value) == pytest.approx(7.0)


def test_compile_failure_retries_another_qualified_matching(monkeypatch: pytest.MonkeyPatch) -> None:
    original = PYTHON_BACKEND.transpile_linear_solve
    attempts = []

    def compile_solve(function: ast.FunctionDef) -> BackendCallable:
        attempts.append(function)
        if len(attempts) == 1:
            message = "qualified formulation refused"
            raise BiltUnsupportedNodeError(message)
        return original(function)

    monkeypatch.setattr(PYTHON_BACKEND, "transpile_linear_solve", compile_solve)
    assert _crossing().evaluate({"inner.p": 10.0, "q": 2.0}, "inner.x", PYTHON_BACKEND) == pytest.approx(6.0)
    assert len(attempts) == len((balance, feedback))


def test_all_qualified_rejections_terminate_and_a_new_request_can_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    implementation = _crossing()
    attempts = []

    def refuse(function: ast.FunctionDef) -> typing.Never:
        attempts.append(function)
        message = "qualified formulations refused"
        raise BiltUnsupportedNodeError(message)

    with monkeypatch.context() as patch:
        patch.setattr(PYTHON_BACKEND, "transpile_linear_solve", refuse)
        with pytest.raises(BiltExecutionPlanError, match=r"inner.balance.*qualified formulations refused"):
            implementation.evaluate({"inner.p": 10.0, "q": 2.0}, "inner.x", PYTHON_BACKEND)
    assert len(attempts) == len((balance, feedback))
    assert implementation.evaluate({"inner.p": 10.0, "q": 2.0}, "inner.x", PYTHON_BACKEND) == pytest.approx(6.0)


def test_unexpected_compiler_errors_are_not_treated_as_alternative_routes(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(_: ast.FunctionDef) -> typing.Never:
        message = "compiler bug"
        raise RuntimeError(message)

    monkeypatch.setattr(PYTHON_BACKEND, "transpile_linear_solve", broken)
    with pytest.raises(RuntimeError, match="compiler bug"):
        _crossing().evaluate({"inner.p": 10.0, "q": 2.0}, "inner.x", PYTHON_BACKEND)


def test_tensor_cross_boundary_blocks_remain_deferred(backend: Backend) -> None:
    with pytest.raises(BiltExecutionPlanError, match="require scalar inputs and unknowns"):
        _crossing().evaluate({"inner.p": tensor([10.0, 12.0], over=dim("time")), "q": 2.0}, "inner.x", backend)


def test_runtime_singularity_does_not_poison_other_input_values(backend: Backend) -> None:
    @equation
    def weighted(x: float, y: float, p: float, a: float) -> bool:
        return a * x + y == p

    @equation
    def other(inner: Pair, q: float) -> bool:
        return inner.x + inner.y == q

    parent = Implementation(Pair, [other], internal_implementations={"inner": Implementation(Pair, [weighted])})
    values: dict[str, object] = {"inner.p": 10.0, "inner.a": 2.0, "q": 6.0}
    assert parent.evaluate(values, "inner.x", backend) == pytest.approx(4.0)
    error = jax.errors.JaxRuntimeError if backend is JAX_BACKEND else BiltLinearSolveError
    with pytest.raises(error, match="singular"):
        parent.evaluate(values | {"inner.a": 1.0}, "inner.x", backend)
    assert parent.evaluate(values, "inner.x", backend) == pytest.approx(4.0)


def test_unrelated_underdetermined_rows_do_not_enter_the_joint_solve(backend: Backend) -> None:
    @equation
    def surplus(u: float, v: float, inner: Pair) -> bool:
        return u + v == inner.x

    parent = Implementation(
        Pair, [feedback, surplus], internal_implementations={"inner": Implementation(Pair, [balance])}
    )
    assert parent.evaluate({"inner.p": 10.0, "q": 2.0}, "inner.x", backend) == pytest.approx(6.0)


@pytest.mark.parametrize(
    ("output", "expected"), (("left.x", 6.0), ("left.y", 4.0), ("right.x", 2.0), ("right.y", 12.0))
)
def test_joint_solve_keeps_sibling_instances_of_one_implementation_distinct(
    backend: Backend, output: str, expected: float
) -> None:
    @equation
    def first(left: Pair, right: Pair, q: float) -> bool:
        return left.y - right.x == q

    @equation
    def second(left: Pair, right: Pair) -> bool:
        return right.y == 2 * left.x

    child = Implementation(Pair, [balance])
    parent = Implementation(Pair, [first, second], internal_implementations={"left": child, "right": child})
    assert parent.evaluate({"left.p": 10.0, "right.p": 14.0, "q": 2.0}, output, backend) == pytest.approx(expected)


def test_only_the_coupled_region_is_assembled_as_a_matrix(monkeypatch: pytest.MonkeyPatch) -> None:
    original = np.linalg.solve
    shapes = []

    def solve(matrix: np.ndarray, rhs: np.ndarray) -> np.ndarray:
        shapes.append(matrix.shape)
        return original(matrix, rhs)

    monkeypatch.setattr(np.linalg, "solve", solve)

    @equation
    def expose(x: float, inner: Pair) -> bool:
        return x == inner.x

    @transform
    def answer(x: float) -> float:
        return x**2

    parent = Implementation(
        Pair, [feedback, expose, answer], internal_implementations={"inner": Implementation(Pair, [balance])}
    )
    assert parent.evaluate({"inner.p": 10.0, "q": 2.0}, "answer", PYTHON_BACKEND) == pytest.approx(36.0)
    assert shapes == [(2, 2)]


def test_nonlinear_parent_child_feedback_is_still_rejected_before_compilation(monkeypatch: pytest.MonkeyPatch) -> None:
    @equation
    def nonlinear(inner: Pair, q: float) -> bool:
        return inner.x * inner.y == q

    def unexpected(_: object) -> typing.Never:
        message = "Nonlinear cross-boundary blocks must not reach linear compilation."
        raise AssertionError(message)

    monkeypatch.setattr(PYTHON_BACKEND, "transpile_linear_solve", unexpected)
    parent = Implementation(Pair, [nonlinear], internal_implementations={"inner": Implementation(Pair, [balance])})
    with pytest.raises(BiltExecutionPlanError, match="Execution of coupled blocks is not yet supported"):
        parent.evaluate({"inner.p": 10.0, "q": 2.0}, "inner.x", PYTHON_BACKEND)


def test_programming_error_in_an_ordered_transform_is_not_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    original = PYTHON_BACKEND.transpile

    def compile_step(function: ast.FunctionDef) -> BackendCallable:
        if function.name == "answer":
            message = "ordered compiler bug"
            raise RuntimeError(message)
        return original(function)

    monkeypatch.setattr(PYTHON_BACKEND, "transpile", compile_step)

    @equation
    def expose(x: float, inner: Pair) -> bool:
        return x == inner.x

    @transform
    def answer(x: float) -> float:
        return x + 1

    parent = Implementation(
        Pair, [feedback, expose, answer], internal_implementations={"inner": Implementation(Pair, [balance])}
    )
    with pytest.raises(RuntimeError, match="ordered compiler bug"):
        parent.evaluate({"inner.p": 10.0, "q": 2.0}, "answer", PYTHON_BACKEND)


def test_unrelated_tensor_input_does_not_add_axes_to_a_scalar_block(backend: Backend) -> None:
    result = _crossing().evaluate(
        {"inner.p": 10.0, "q": 2.0, "raw": tensor([1.0, 2.0], over=dim("unrelated"))}, "inner.x", backend
    )
    assert np.asarray(result).shape == ()
    assert result == pytest.approx(6.0)


def test_constant_cross_boundary_block_has_no_external_inputs(backend: Backend) -> None:
    total_value = 10.0
    difference_value = 2.0

    @equation
    def constant(x: float, y: float) -> bool:
        return x + y == total_value

    @equation
    def relation(inner: Pair) -> bool:
        return inner.x - inner.y == difference_value

    parent = Implementation(Pair, [relation], internal_implementations={"inner": Implementation(Pair, [constant])})
    assert parent.evaluate({}, "inner.x", backend) == pytest.approx(6.0)

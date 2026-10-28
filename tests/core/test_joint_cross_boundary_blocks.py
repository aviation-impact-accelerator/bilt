import dataclasses
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
from bilt.core._structural_analysis import _realize_linear as linear_realization
from bilt.core._structural_analysis._causalization_block_solver._backend import CausalizationStructuralAnalyzer
from bilt.core._structural_analysis._causalization_block_solver._errors import BiltExecutionPlanError
from bilt.core._sub_system import SubSystemConstituent
from bilt.core._system import _implementation as implementation_module
from bilt.errors import BiltLinearSolveError
from bilt.symbolic._dimensions import coordinate_axis
from bilt.units.dimensions import CURRENT, RESISTANCE, VOLTAGE
from bilt.units.units import ampere, ohm, volt

if typing.TYPE_CHECKING:
    import ast

    from bilt._backend import Backend, BackendCallable

TIME = dim("time")
SITE = dim("site")
BRANCH = dim("branch")
OverTime = typing.Annotated[Tensor, domain(TIME)]
Shared = typing.Annotated[Tensor, Domain.scalar()]
Grid = typing.Annotated[Tensor, domain(SITE, TIME)]
PerSite = typing.Annotated[Tensor, domain(SITE)]
BranchCurrent = typing.Annotated[Tensor, CURRENT, domain(BRANCH)]
SharedVoltage = typing.Annotated[Quantity, VOLTAGE, Domain.scalar()]
VoltageArray = typing.Annotated[Tensor, VOLTAGE]
ResistanceArray = typing.Annotated[Tensor, RESISTANCE]
Resistance = typing.Annotated[Quantity, RESISTANCE]


@pytest.fixture(autouse=True)
def _use_scc(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(implementation_module, "ExhaustiveStructuralAnalyser", CausalizationStructuralAnalyzer)


@pytest.fixture(params=(PYTHON_BACKEND, JAX_BACKEND), ids=("python", "jax"))
def backend(request: pytest.FixtureRequest) -> Backend:
    return request.param


@system
class Ports(metaclass=Interface):
    x: Tensor = field()
    y: float | Tensor = field()
    p: Tensor = field()
    q: float | Tensor = field()
    weights: Tensor = field()
    offset: float = field()


@equation
def boundary(p: Tensor, inner: Ports) -> bool:
    return inner.x + inner.y == p


@equation
def supply_q(q: Tensor, inner: Ports) -> bool:
    return inner.q == q


def _model(reduction: str = "sum") -> Implementation:
    @equation
    def aggregate(x: OverTime, y: Shared, q: float) -> bool:
        return (0.5 * x.sum(TIME) if reduction == "sum" else x.mean(TIME)) - y == q

    return Implementation(
        Ports, [boundary, supply_q], internal_implementations={"inner": Implementation(Ports, [aggregate])}
    )


@pytest.mark.parametrize("reduction", ("sum", "mean"))
@pytest.mark.parametrize("output", ("inner.x", "inner.y"))
def test_child_reduction_and_parent_balance_solve_tensor_and_shared_unknowns(
    backend: Backend, reduction: str, output: str
) -> None:
    result = _model(reduction).evaluate({"p": tensor([2.0, 4.0, 6.0], over=TIME), "q": 1.0}, output, backend)
    shared = 2.0 if reduction == "sum" else 1.5
    if output == "inner.x":
        assert isinstance(result, Tensor)
        assert result.domain == domain(TIME)
        np.testing.assert_allclose(result.values, np.array([2.0, 4.0, 6.0]) - shared, atol=1e-5)
    else:
        assert not isinstance(result, Tensor)
        assert np.asarray(result).shape == ()
        assert result == pytest.approx(shared, abs=1e-5)


def test_one_shared_unknown_adds_one_column_to_the_joint_matrix(monkeypatch: pytest.MonkeyPatch) -> None:
    original = np.linalg.solve
    shapes = []

    def solve(matrix: np.ndarray, rhs: np.ndarray) -> np.ndarray:
        shapes.append(matrix.shape)
        return original(matrix, rhs)

    monkeypatch.setattr(np.linalg, "solve", solve)
    result = _model().evaluate({"p": tensor([2.0, 4.0, 6.0], over=TIME), "q": 1.0}, "inner.y", PYTHON_BACKEND)
    assert result == pytest.approx(2.0)
    assert shapes == [(4, 4)]


@pytest.mark.parametrize("output", ("inner.x", "inner.y"))
@pytest.mark.parametrize("reverse", (False, True))
def test_shared_unknown_can_vary_over_a_subset_of_the_tensor_axes(
    backend: Backend, output: str, *, reverse: bool
) -> None:
    @equation
    def aggregate(x: Grid, y: PerSite, q: Tensor) -> bool:
        return 0.5 * x.sum(TIME) - y == q

    parent = Implementation(
        Ports, [boundary, supply_q], internal_implementations={"inner": Implementation(Ports, [aggregate])}
    )
    values = np.array([[2.0, 4.0, 6.0], [4.0, 8.0, 12.0]])
    axes = (TIME, SITE) if reverse else (SITE, TIME)
    with EvaluationContext() as evaluation:
        for axis in axes:
            evaluation.domains.bind(coordinate_axis(axis, (0, 1, 2) if axis == TIME else (0, 1)))
        result = parent.evaluate(
            {"p": tensor(values, over=(SITE, TIME)), "q": tensor([1.0, 2.0], over=SITE)}, output, backend
        )
    shared = (0.5 * values.sum(axis=1) - np.array([1.0, 2.0])) / 2.5
    assert isinstance(result, Tensor)
    if output == "inner.x":
        assert result.domain == domain(*axes)
        expected = values - shared[:, None]
        np.testing.assert_allclose(result.values, expected.T if reverse else expected, atol=1e-5)
    else:
        assert result.domain == domain(SITE)
        np.testing.assert_allclose(result.values, shared, atol=1e-5)


def test_broadcasting_inside_a_reduction_is_preserved_across_qualified_rows(backend: Backend) -> None:
    @equation
    def aggregate(x: OverTime, y: Shared, weights: Tensor, offset: float, q: float) -> bool:
        return (weights * x + y + offset).sum(TIME) == q

    parent = Implementation(Ports, [boundary], internal_implementations={"inner": Implementation(Ports, [aggregate])})
    result = parent.evaluate(
        {
            "p": tensor([2.0, 4.0, 6.0], over=TIME),
            "inner.weights": tensor([1.0, 2.0, 3.0], over=TIME),
            "inner.offset": 1.0,
            "inner.q": 25.0,
        },
        "inner.y",
        backend,
    )
    assert result == pytest.approx(2.0, abs=1e-5)


@pytest.mark.parametrize("output", ("inner.current", "inner.voltage"))
def test_shared_bus_voltage_and_branch_currents_retain_units(backend: Backend, output: str) -> None:
    @system
    class Bus(metaclass=Interface):
        current: BranchCurrent = field()
        voltage: SharedVoltage = field()
        resistance: ResistanceArray = field()
        load: Resistance = field()

    @system
    class Sources(metaclass=Interface):
        emf: VoltageArray = field()

    @equation
    def load_balance(current: BranchCurrent, voltage: SharedVoltage, load: Resistance) -> bool:
        return voltage == load * current.sum(BRANCH)

    @equation
    def branch_balance(emf: VoltageArray, inner: Bus) -> bool:
        return inner.resistance * inner.current + inner.voltage == emf

    parent = Implementation(
        Sources, [branch_balance], internal_implementations={"inner": Implementation(Bus, [load_balance])}
    )
    result = parent.evaluate(
        {
            "emf": tensor([10.0, 12.0, 14.0], over=BRANCH, unit=volt),
            "inner.resistance": tensor([1.0, 1.0, 1.0], over=BRANCH, unit=ohm),
            "inner.load": 0.5 * ohm,
        },
        output,
        backend,
    )
    if output == "inner.current":
        assert isinstance(result, Tensor)
        assert result.domain == domain(BRANCH)
        assert result.unit == ampere
        np.testing.assert_allclose(result.values, [2.8, 4.8, 6.8], atol=1e-5)
    else:
        assert isinstance(result, Quantity)
        assert result.unit == volt
        assert float(result.value) == pytest.approx(7.2, abs=1e-5)


@pytest.mark.parametrize("output", ("x", "y"))
def test_reduction_can_belong_to_the_parent_using_tensor_aliases(backend: Backend, output: str) -> None:
    @equation
    def child_balance(x: OverTime, y: Shared, p: Tensor) -> bool:
        return x + y == p

    @equation
    def expose_x(x: OverTime, inner: Ports) -> bool:
        return x == inner.x

    @equation
    def expose_y(y: Shared, inner: Ports) -> bool:
        return y == inner.y

    @equation
    def supply_p(p: Tensor, inner: Ports) -> bool:
        return inner.p == p

    @equation
    def parent_aggregate(x: OverTime, y: Shared, q: float) -> bool:
        return 0.5 * x.sum(TIME) - y == q

    parent = Implementation(
        Ports,
        [expose_x, expose_y, supply_p, parent_aggregate],
        internal_implementations={"inner": Implementation(Ports, [child_balance])},
    )
    result = parent.evaluate({"p": tensor([2.0, 4.0, 6.0], over=TIME), "q": 1.0}, output, backend)
    if output == "x":
        assert isinstance(result, Tensor)
        np.testing.assert_allclose(result.values, [0.0, 2.0, 4.0], atol=1e-5)
    else:
        assert result == pytest.approx(2.0, abs=1e-5)


@pytest.mark.parametrize("output", ("inner.x", "inner.y"))
@pytest.mark.parametrize("selection", ("isel", "sel"))
def test_literal_selection_connects_coordinates_in_a_qualified_joint_block(
    backend: Backend, output: str, selection: str
) -> None:
    @equation
    def selected(x: OverTime, y: OverTime, q: Tensor) -> bool:
        return x - 0.25 * getattr(y, selection)(TIME, 0 if selection == "isel" else "reference") == q

    parent = Implementation(Ports, [boundary], internal_implementations={"inner": Implementation(Ports, [selected])})
    with EvaluationContext() as evaluation:
        evaluation.domains.bind(coordinate_axis(TIME, ("reference", "other")))
        result = parent.evaluate(
            {"p": tensor([2.0, 4.0], over=TIME), "inner.q": tensor([0.0, 1.0], over=TIME)}, output, backend
        )
    assert isinstance(result, Tensor)
    assert result.domain == domain(TIME)
    np.testing.assert_allclose(result.values, [0.4, 1.4] if output == "inner.x" else [1.6, 2.6], atol=1e-5)


def test_missing_unknown_hints_are_rejected_without_domain_inference(backend: Backend) -> None:
    @equation
    def unhinted(x: Tensor, y: Tensor, q: float) -> bool:
        return 0.5 * x.sum(TIME) - y == q

    parent = Implementation(
        Ports, [boundary, supply_q], internal_implementations={"inner": Implementation(Ports, [unhinted])}
    )
    with pytest.raises(BiltExecutionPlanError, match=r"explicit Domain hints.*inner.x.*inner.y"):
        parent.evaluate({"p": tensor([2.0, 4.0, 6.0], over=TIME), "q": 1.0}, "inner.x", backend)


@pytest.mark.parametrize("output", ("inner.x", "inner.y"))
def test_empty_tensor_still_leaves_one_shared_scalar_to_solve(backend: Backend, output: str) -> None:
    result = _model().evaluate({"p": tensor([], over=TIME), "q": 3.0}, output, backend)
    if output == "inner.x":
        assert isinstance(result, Tensor)
        assert result.domain == domain(TIME)
        assert result.values.shape == (0,)
    else:
        assert result == pytest.approx(-3.0)


def test_cached_joint_function_handles_different_extents_without_bilt_recompilation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = PYTHON_BACKEND.transpile_linear_solve
    calls = []

    def compile_solve(function: ast.FunctionDef) -> BackendCallable:
        calls.append(function)
        return original(function)

    monkeypatch.setattr(PYTHON_BACKEND, "transpile_linear_solve", compile_solve)
    parent = _model()
    for values in ([2.0, 4.0, 6.0], [2.0, 4.0]):
        result = parent.evaluate({"p": tensor(values, over=TIME), "q": 1.0}, "inner.y", PYTHON_BACKEND)
        expected = (0.5 * sum(values) - 1) / (0.5 * len(values) + 1)
        assert result == pytest.approx(expected)
    assert len(calls) == 1


def test_compile_failure_retries_another_joint_formulation(monkeypatch: pytest.MonkeyPatch) -> None:
    original = PYTHON_BACKEND.transpile_linear_solve
    calls = []

    def compile_solve(function: ast.FunctionDef) -> BackendCallable:
        calls.append(function)
        if len(calls) == 1:
            message = "joint formulation refused"
            raise BiltUnsupportedNodeError(message)
        return original(function)

    monkeypatch.setattr(PYTHON_BACKEND, "transpile_linear_solve", compile_solve)
    result = _model().evaluate({"p": tensor([2.0, 4.0, 6.0], over=TIME), "q": 1.0}, "inner.y", PYTHON_BACKEND)
    assert result == pytest.approx(2.0)
    assert len(calls) == len(("balance", "aggregate"))


def test_jax_joint_cross_boundary_solve_is_differentiable() -> None:
    parent = _model()
    with EvaluationContext() as evaluation:
        evaluation.domains.bind(tensor([2.0, 4.0, 6.0], over=TIME))
        function = parent.get_evaluation(
            EvaluationRequest(output="inner.y", input_domains={"p": domain(TIME), "q": Domain.scalar()}), JAX_BACKEND
        )
    derivative = jax.jacrev(lambda p: function.callable(p=p, q=1.0))(jnp.array([2.0, 4.0, 6.0]))
    np.testing.assert_allclose(derivative, [[0.2, 0.2, 0.2]], atol=1e-5)


def _nested_model(depth: int) -> Implementation:
    @equation
    def aggregate(x: OverTime, y: Shared, q: float) -> bool:
        return 0.5 * x.sum(TIME) - y == q

    @equation
    def expose_x(x: OverTime, inner: Ports) -> bool:
        return x == inner.x

    @equation
    def expose_y(y: Shared, inner: Ports) -> bool:
        return y == inner.y

    parent = Implementation(
        Ports,
        [boundary, supply_q, expose_x, expose_y],
        internal_implementations={"inner": Implementation(Ports, [aggregate])},
    )
    for _ in range(depth):

        @equation
        def supply_p(p: Tensor, inner: Ports) -> bool:
            return inner.p == p

        parent = Implementation(
            Ports, [supply_p, supply_q, expose_x, expose_y], internal_implementations={"inner": parent}
        )
    return parent


@pytest.mark.parametrize("depth", (1, 2))
@pytest.mark.parametrize("output", ("x", "y"))
def test_joint_layouts_and_output_domains_survive_nested_boundaries(backend: Backend, depth: int, output: str) -> None:
    result = _nested_model(depth).evaluate({"p": tensor([2.0, 4.0, 6.0], over=TIME), "q": 1.0}, output, backend)
    if output == "x":
        assert isinstance(result, Tensor)
        assert result.domain == domain(TIME)
        np.testing.assert_allclose(result.values, [0.0, 2.0, 4.0], atol=1e-5)
    else:
        assert not isinstance(result, Tensor)
        assert result == pytest.approx(2.0, abs=1e-5)


def test_shared_implementation_instances_keep_separate_joint_systems(
    backend: Backend, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = np.linalg.solve
    shapes = []

    def solve(matrix: np.ndarray, rhs: np.ndarray) -> np.ndarray:
        shapes.append(matrix.shape)
        return original(matrix, rhs)

    monkeypatch.setattr(np.linalg, "solve", solve)

    @equation
    def combine(y: Shared, left: Ports, right: Ports) -> bool:
        return y == left.y + right.y

    shared = _nested_model(0)
    parent = Implementation(Ports, [combine], internal_implementations={"left": shared, "right": shared})
    result = parent.evaluate(
        {
            "left.p": tensor([2.0, 4.0, 6.0], over=TIME),
            "left.q": 1.0,
            "right.p": tensor([4.0, 8.0, 12.0], over=TIME),
            "right.q": 2.0,
        },
        "y",
        backend,
    )
    assert result == pytest.approx(6.0, abs=1e-5)
    if backend is PYTHON_BACKEND:
        assert shapes == [(4, 4), (4, 4)]


def test_required_inputs_can_describe_a_child_with_a_joint_cross_boundary_route(backend: Backend) -> None:
    child = SubSystemConstituent("module", _nested_model(0))
    assert child.required_inputs_for("module.y", backend, within={"module.p", "module.q"}) == frozenset({"p", "q"})


def test_partial_hints_do_not_invoke_cross_boundary_domain_inference(monkeypatch: pytest.MonkeyPatch) -> None:
    def unexpected(*_: object, **__: object) -> typing.Never:
        message = "Qualified joint plans must request missing hints rather than infer them in this PR."
        raise AssertionError(message)

    monkeypatch.setattr(linear_realization, "infer_linear_domains", unexpected)
    assert _model().evaluate(
        {"p": tensor([2.0, 4.0, 6.0], over=TIME), "q": 1.0}, "inner.y", PYTHON_BACKEND
    ) == pytest.approx(2.0)

    @equation
    def partial(x: OverTime, y: Tensor, q: float) -> bool:
        return 0.5 * x.sum(TIME) - y == q

    parent = Implementation(
        Ports, [boundary, supply_q], internal_implementations={"inner": Implementation(Ports, [partial])}
    )
    with pytest.raises(BiltExecutionPlanError, match=r"explicit Domain hints.*inner.y"):
        parent.evaluate({"p": tensor([2.0, 4.0, 6.0], over=TIME), "q": 1.0}, "inner.y", PYTHON_BACKEND)


@pytest.mark.parametrize("output", ("inner.x", "inner.y"))
def test_both_unknowns_can_be_shared_when_known_weights_supply_the_reduction_axis(
    backend: Backend, output: str
) -> None:
    @equation
    def aggregate(x: Shared, y: Shared, weights: Tensor, q: float) -> bool:
        return (weights * x + y).sum(TIME) == q

    parent = Implementation(Ports, [boundary], internal_implementations={"inner": Implementation(Ports, [aggregate])})
    result = parent.evaluate(
        {"p": 6.0, "inner.weights": tensor([1.0, 2.0, 3.0], over=TIME), "inner.q": 24.0}, output, backend
    )
    assert not isinstance(result, Tensor)
    assert result == pytest.approx(2.0 if output == "inner.x" else 4.0, abs=1e-5)


def test_coordinate_labels_and_extents_select_the_correct_cached_joint_plan(backend: Backend) -> None:
    @equation
    def selected(x: OverTime, y: Shared, q: float) -> bool:
        return x.sel(TIME, "reference") - y == q

    parent = Implementation(Ports, [boundary], internal_implementations={"inner": Implementation(Ports, [selected])})
    for coordinates, values, expected in (
        (("reference", "other"), [2.0, 4.0], 1.0),
        (("other", "reference"), [2.0, 4.0], 2.0),
        (("other", "third", "reference"), [2.0, 4.0, 6.0], 3.0),
    ):
        with EvaluationContext() as evaluation:
            evaluation.domains.bind(coordinate_axis(TIME, coordinates))
            result = parent.evaluate({"p": tensor(values, over=TIME), "inner.q": 0.0}, "inner.y", backend)
        assert result == pytest.approx(expected, abs=1e-5)


def test_missing_joint_backend_helper_rejects_and_does_not_poison_a_later_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = _model()
    with monkeypatch.context() as patch:
        patch.setattr(
            PYTHON_BACKEND,
            "intrinsic_bindings",
            dataclasses.replace(PYTHON_BACKEND.intrinsic_bindings, linear_zero=None),
        )
        with pytest.raises(BiltExecutionPlanError, match="does not support joint tensor linear blocks"):
            parent.evaluate({"p": tensor([2.0, 4.0, 6.0], over=TIME), "q": 1.0}, "inner.y", PYTHON_BACKEND)
    assert parent.evaluate(
        {"p": tensor([2.0, 4.0, 6.0], over=TIME), "q": 1.0}, "inner.y", PYTHON_BACKEND
    ) == pytest.approx(2.0)


def test_runtime_joint_singularity_does_not_poison_other_coefficient_values(backend: Backend) -> None:
    @equation
    def aggregate(x: OverTime, y: Shared, coefficient: float, q: float) -> bool:
        return coefficient * x.sum(TIME) - y == q

    parent = Implementation(
        Ports, [boundary, supply_q], internal_implementations={"inner": Implementation(Ports, [aggregate])}
    )
    values: dict[str, object] = {"p": tensor([2.0, 4.0], over=TIME), "q": 1.0, "inner.coefficient": 0.5}
    assert parent.evaluate(values, "inner.y", backend) == pytest.approx(1.0, abs=1e-5)
    error = jax.errors.JaxRuntimeError if backend is JAX_BACKEND else BiltLinearSolveError
    with pytest.raises(error, match="singular"):
        parent.evaluate(values | {"inner.coefficient": -0.5}, "inner.y", backend)
    assert parent.evaluate(values, "inner.y", backend) == pytest.approx(1.0, abs=1e-5)

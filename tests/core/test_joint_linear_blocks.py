import dataclasses
import typing

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bilt import Implementation, Interface, Tensor, dim, equation, field, system, tensor, transform
from bilt._backend._jax._backend import JAX_BACKEND
from bilt._backend._python._backend import PYTHON_BACKEND
from bilt.core._evaluation_context import EvaluationContext
from bilt.core._evaluation_request import EvaluationRequest
from bilt.core._structural_analysis._causalization_block_solver._backend import CausalizationStructuralAnalyzer
from bilt.core._structural_analysis._causalization_block_solver._errors import BiltExecutionPlanError
from bilt.core._system import _implementation as implementation_module
from bilt.core._system._lowering_and_lifting import lower_inputs
from bilt.core._tensors._tensor_alignment import generate_alignment_plan
from bilt.errors import BiltLinearSolveError
from bilt.symbolic._domains import Domain, domain
from bilt.units import Quantity
from bilt.units.dimensions import CURRENT, LENGTH, RESISTANCE, VOLTAGE
from bilt.units.units import ampere, kilometer, meter, ohm, volt

if typing.TYPE_CHECKING:
    from bilt._backend import Backend, BackendCase
    from bilt.core._model_constituent import BiltModelConstituent


TIME = dim("time")
SITE = dim("site")
BRANCH = dim("branch")
X = typing.Annotated[Tensor, domain(TIME)]
Scalar = typing.Annotated[Tensor, Domain.scalar()]
Grid = typing.Annotated[Tensor, domain(SITE, TIME)]
PerSite = typing.Annotated[Tensor, domain(SITE)]
LengthArray = typing.Annotated[Tensor, LENGTH]
LengthGrid = typing.Annotated[Tensor, LENGTH, domain(TIME)]
LengthScalar = typing.Annotated[Quantity, LENGTH, Domain.scalar()]
ReverseGrid = typing.Annotated[Tensor, domain(TIME, SITE)]
CurrentArray = typing.Annotated[Tensor, CURRENT]
BranchCurrent = typing.Annotated[Tensor, CURRENT, domain(BRANCH)]
SharedVoltage = typing.Annotated[Quantity, VOLTAGE, Domain.scalar()]
VoltageArray = typing.Annotated[Tensor, VOLTAGE]
Resistance = typing.Annotated[Quantity, RESISTANCE]


@pytest.fixture(autouse=True)
def _use_scc(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(implementation_module, "ExhaustiveStructuralAnalyser", CausalizationStructuralAnalyzer)


@pytest.fixture(params=(PYTHON_BACKEND, JAX_BACKEND), ids=("python", "jax"))
def backend(request: pytest.FixtureRequest) -> Backend:
    return request.param


@equation
def balance(x: X, y: Scalar, p: Tensor) -> bool:
    return x + y == p


@equation
def aggregate(x: X, y: Scalar, q: float) -> bool:
    return 0.5 * x.sum(TIME) - y == q


@system
class Pair(metaclass=Interface):
    x: Tensor = field()
    y: Tensor = field()
    p: Tensor = field()
    q: float = field()
    a: float | Tensor = field()


def _evaluate(
    operations: typing.Collection[BiltModelConstituent], values: dict[str, object], output: str, backend: Backend
) -> tuple[np.ndarray, BackendCase]:
    with EvaluationContext() as evaluation:
        evaluation.domains.bind_all(values.values())
        inputs = {
            name: value.domain if isinstance(value, Tensor) else Domain.scalar() for name, value in values.items()
        }
        case = CausalizationStructuralAnalyzer(operations, backend).analyse(
            EvaluationRequest(output=output, input_domains=inputs)
        )
        aligned = backend.apply_alignment(lower_inputs(values), generate_alignment_plan(values))
        result = backend.chain_case(case).evaluate(aligned)
        return np.asarray(result), case


@pytest.mark.parametrize("output", ("x", "y"))
def test_shared_scalar_and_tensor_unknown_form_one_joint_block(backend: Backend, output: str) -> None:
    with backend, Implementation(Pair, [balance, aggregate]):
        result = getattr(Pair(p=tensor([2.0, 4.0, 6.0], over=TIME), q=1.0), output)
    if output == "x":
        assert isinstance(result, Tensor)
        assert result.domain == domain(TIME)
        np.testing.assert_allclose(result.values, [0, 2, 4], atol=1e-5)
    else:
        assert np.asarray(result).ndim == 0
        np.testing.assert_allclose(result, 2, atol=1e-5)


def test_joint_matrix_has_one_column_for_the_shared_scalar(monkeypatch: pytest.MonkeyPatch) -> None:
    original = np.linalg.solve
    shapes = []

    def solve(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        shapes.append(a.shape)
        return original(a, b)

    monkeypatch.setattr(np.linalg, "solve", solve)
    result, _ = _evaluate(
        [balance, aggregate], {"p": tensor([2.0, 4.0, 6.0], over=TIME), "q": 1.0}, "x", PYTHON_BACKEND
    )
    np.testing.assert_allclose(result, [0, 2, 4], atol=1e-10)
    assert shapes == [(4, 4)]


def test_mean_couples_coordinates_within_the_block(backend: Backend) -> None:
    @equation
    def first(x: X, y: X, p: Tensor) -> bool:
        return x + y == p

    @equation
    def second(x: X, y: X, q: Tensor) -> bool:
        return x - 0.5 * y.mean(TIME) == q

    result, _ = _evaluate(
        [first, second],
        {"p": tensor([2.0, 4.0, 6.0], over=TIME), "q": tensor([0.0, 1.0, 2.0], over=TIME)},
        "x",
        backend,
    )
    np.testing.assert_allclose(result, [1, 2, 3], rtol=1e-5)


def test_cached_joint_solve_accepts_different_dimension_extents(backend: Backend) -> None:
    implementation = Implementation(Pair, [balance, aggregate])
    with backend, implementation:
        first = Pair(p=tensor([2.0, 4.0, 6.0], over=TIME), q=1.0).x
        second = Pair(p=tensor([2.0, 4.0], over=TIME), q=1.0).x
    np.testing.assert_allclose(first.values, [0, 2, 4], atol=1e-5)
    np.testing.assert_allclose(second.values, [1, 3], atol=1e-5)


def test_missing_operator_domain_hints_are_rejected(backend: Backend) -> None:
    @equation
    def first(x: Tensor, y: Tensor, p: Tensor) -> bool:
        return x + y == p

    @equation
    def second(x: Tensor, y: Tensor, q: float) -> bool:
        return x.sum(TIME) - y == q

    with pytest.raises(BiltExecutionPlanError, match="explicit Domain hints"):
        _evaluate([first, second], {"p": tensor([2.0, 4.0], over=TIME), "q": 1.0}, "x", backend)


def test_inconsistent_scalar_equation_counts_are_rejected_before_execution(backend: Backend) -> None:
    @equation
    def second(x: X, y: Scalar, q: float) -> bool:
        return x - y == q

    with pytest.raises(BiltExecutionPlanError, match="scalar equations"):
        _evaluate([balance, second], {"p": tensor([2.0, 4.0], over=TIME), "q": 1.0}, "x", backend)


def test_shared_unknown_can_vary_over_a_subset_of_dimensions(backend: Backend) -> None:

    @equation
    def first(x: Grid, y: PerSite, p: Tensor) -> bool:
        return x + y == p

    @equation
    def second(x: Grid, y: PerSite, q: Tensor) -> bool:
        return 0.5 * x.sum(TIME) - y == q

    p = np.array([[2.0, 4.0, 6.0], [4.0, 8.0, 12.0]])
    q = np.array([1.0, 2.0])
    result, _ = _evaluate([first, second], {"p": tensor(p, over=(SITE, TIME)), "q": tensor(q, over=SITE)}, "x", backend)
    np.testing.assert_allclose(result, p - np.array([2.0, 4.0])[:, None], atol=1e-5)


def test_conflicting_unknown_domain_hints_are_rejected(backend: Backend) -> None:
    @equation
    def second(x: X, y: X, q: float) -> bool:
        return x - y == q

    with pytest.raises(BiltExecutionPlanError, match="Conflicting linear-block domain hints"):
        _evaluate([balance, second], {"p": tensor([2.0, 4.0], over=TIME), "q": 1.0}, "x", backend)


def test_unknown_dimensions_cannot_be_invented_from_unbound_axes(backend: Backend) -> None:
    with pytest.raises(BiltExecutionPlanError, match="dimensions present in the block inputs"):
        _evaluate([balance, aggregate], {"p": tensor([2.0, 4.0], over=SITE), "q": 1.0}, "x", backend)


@pytest.mark.parametrize("operation", ("min", "max", "prod"))
def test_nonlinear_reductions_of_unknowns_remain_unsupported(backend: Backend, operation: str) -> None:
    @equation
    def second(x: X, y: Scalar, q: float) -> bool:
        return getattr(x, operation)(TIME) - y == q

    with pytest.raises(BiltExecutionPlanError, match="coupled blocks"):
        _evaluate([balance, second], {"p": tensor([2.0, 4.0], over=TIME), "q": 1.0}, "x", backend)


@pytest.mark.parametrize("output", ("x", "y"))
def test_joint_solve_retains_units_and_distinct_output_domains(backend: Backend, output: str) -> None:
    @system
    class Lengths(metaclass=Interface):
        x: LengthArray = field()
        y: LengthScalar = field()
        p: LengthArray = field()
        q: LengthScalar = field()

    @equation
    def first(x: LengthGrid, y: LengthScalar, p: LengthArray) -> bool:
        return x + y == p

    @equation
    def second(x: LengthGrid, y: LengthScalar, q: LengthScalar) -> bool:
        return 0.5 * x.sum(TIME) - y == q

    with backend, Implementation(Lengths, [first, second]):
        result = getattr(Lengths(p=tensor([0.002, 0.004, 0.006], over=TIME, unit=kilometer), q=1.0 * meter), output)
    assert result.unit == meter
    if output == "x":
        assert isinstance(result, Tensor)
        assert result.domain == domain(TIME)
        np.testing.assert_allclose(result.values, [0, 2, 4], atol=1e-5)
    else:
        assert isinstance(result, Quantity)
        assert float(result.value) == pytest.approx(2.0, abs=1e-5)


def test_named_domain_hints_are_independent_of_axis_declaration_order(backend: Backend) -> None:
    @equation
    def first(x: Grid, y: ReverseGrid, p: Tensor) -> bool:
        return x + y == p

    @equation
    def second(x: ReverseGrid, y: Grid, q: float) -> bool:
        return x - y == q

    p = np.arange(6.0).reshape(2, 3)
    result, _ = _evaluate([first, second], {"p": tensor(p, over=(SITE, TIME)), "q": 0.0}, "x", backend)
    np.testing.assert_allclose(result, p / 2, rtol=1e-5)


def test_backend_without_joint_assembly_support_rejects_the_block(monkeypatch: pytest.MonkeyPatch) -> None:
    bindings = dataclasses.replace(PYTHON_BACKEND.intrinsic_bindings, linear_basis=None)
    monkeypatch.setattr(PYTHON_BACKEND, "intrinsic_bindings", bindings)
    with pytest.raises(BiltExecutionPlanError, match="joint tensor linear blocks"):
        _evaluate([balance, aggregate], {"p": tensor([2.0, 4.0], over=TIME), "q": 1.0}, "x", PYTHON_BACKEND)


def test_jax_joint_solve_remains_differentiable() -> None:
    _, case = _evaluate([balance, aggregate], {"p": tensor([2.0, 4.0, 6.0], over=TIME), "q": 1.0}, "x", JAX_BACKEND)
    compiled = JAX_BACKEND.chain_case(case)
    jacobian = jax.jacrev(lambda p: compiled.callable(p=p, q=1.0))(jnp.array([2.0, 4.0, 6.0]))
    np.testing.assert_allclose(jacobian, np.eye(3) - 0.2 * np.ones((3, 3)), atol=1e-5)


def test_joint_and_batched_blocks_fit_between_ordinary_plan_steps(backend: Backend) -> None:
    @transform
    def p(raw: Tensor) -> Tensor:
        return 2 * raw

    @equation
    def second_total(u: float, v: float, x: float) -> bool:
        return u + v == x

    @equation
    def second_difference(u: float, v: float, y: float) -> bool:
        return u - v == y

    @equation
    def finish(answer: float, u: Tensor) -> bool:
        return answer == u.sum(TIME)

    result, case = _evaluate(
        [p, balance, aggregate, second_total, second_difference, finish],
        {"raw": tensor([1.0, 2.0, 3.0], over=TIME), "q": 1.0},
        "answer",
        backend,
    )
    np.testing.assert_allclose(result, [6.0], atol=1e-5)
    assert case.intermediates[-1].output_domain == Domain.scalar()


@pytest.mark.parametrize("output", ("current", "bus_voltage"))
def test_parallel_sources_share_one_bus_voltage(backend: Backend, output: str) -> None:
    @system
    class Circuit(metaclass=Interface):
        current: CurrentArray = field()
        bus_voltage: SharedVoltage = field()
        source_voltage: VoltageArray = field()
        branch_resistance: Resistance = field()
        load_resistance: Resistance = field()

    @equation
    def branch_balance(
        current: BranchCurrent,
        bus_voltage: SharedVoltage,
        source_voltage: VoltageArray,
        branch_resistance: Resistance,
    ) -> bool:
        return branch_resistance * current + bus_voltage == source_voltage

    @equation
    def load_balance(current: BranchCurrent, bus_voltage: SharedVoltage, load_resistance: Resistance) -> bool:
        return bus_voltage == load_resistance * current.sum(BRANCH)

    with backend, Implementation(Circuit, [branch_balance, load_balance]):
        circuit = Circuit(
            source_voltage=tensor([10.0, 12.0, 14.0], over=BRANCH, unit=volt),
            branch_resistance=1.0 * ohm,
            load_resistance=0.5 * ohm,
        )
        result = getattr(circuit, output)
    if output == "current":
        assert isinstance(result, Tensor)
        assert result.domain == domain(BRANCH)
        assert result.unit == ampere
        np.testing.assert_allclose(result.values, [2.8, 4.8, 6.8], atol=1e-5)
    else:
        assert isinstance(result, Quantity)
        assert result.unit == volt
        assert float(result.value) == pytest.approx(7.2, abs=1e-5)


def test_partial_unknown_domain_hints_are_rejected(backend: Backend) -> None:
    @equation
    def first(x: X, y: Tensor, p: Tensor) -> bool:
        return x + y == p

    @equation
    def second(x: X, y: Tensor, q: float) -> bool:
        return x.sum(TIME) - y == q

    with pytest.raises(BiltExecutionPlanError, match="explicit Domain hints"):
        _evaluate([first, second], {"p": tensor([2.0, 4.0], over=TIME), "q": 1.0}, "x", backend)


def test_singular_joint_system_is_reported(backend: Backend) -> None:
    @equation
    def redundant(x: X, y: Scalar, q: float) -> bool:
        return x.sum(TIME) + 2 * y == q

    with pytest.raises((BiltLinearSolveError, RuntimeError), match="singular"):
        _evaluate([balance, redundant], {"p": tensor([2.0, 4.0], over=TIME), "q": 6.0}, "x", backend)


@pytest.mark.parametrize("bad", (float("nan"), float("inf")))
def test_nonfinite_joint_data_is_reported(backend: Backend, bad: float) -> None:
    with pytest.raises((BiltLinearSolveError, RuntimeError), match="finite"):
        _evaluate([balance, aggregate], {"p": tensor([bad, 4.0, 6.0], over=TIME), "q": 1.0}, "x", backend)


def test_declared_input_domains_must_match_actual_inputs(backend: Backend) -> None:
    @equation
    def first(x: X, y: Scalar, p: PerSite) -> bool:
        return x + y == p

    with pytest.raises(BiltExecutionPlanError, match="does not match its linear-block Domain hint"):
        _evaluate([first, aggregate], {"p": tensor([2.0, 4.0], over=TIME), "q": 1.0}, "x", backend)

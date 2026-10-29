import typing

import numpy as np
import pytest

from bilt import Domain, Implementation, Interface, Quantity, Tensor, dim, domain, equation, field, system, tensor
from bilt._backend._jax._backend import JAX_BACKEND
from bilt._backend._python._backend import PYTHON_BACKEND
from bilt.core._evaluation_context import EvaluationContext
from bilt.core._structural_analysis._causalization_block_solver._backend import CausalizationStructuralAnalyzer
from bilt.core._structural_analysis._causalization_block_solver._errors import BiltExecutionPlanError
from bilt.core._system import _implementation as implementation_module
from bilt.symbolic._dimensions import coordinate_axis
from bilt.units.dimensions import CURRENT, RESISTANCE, VOLTAGE
from bilt.units.units import ampere, ohm, volt

if typing.TYPE_CHECKING:
    from bilt._backend import Backend
    from bilt.core._model_constituent import BiltModelConstituent

TIME = dim("time")
SITE = dim("site")
BRANCH = dim("branch")
Shared = typing.Annotated[float, Domain.scalar()]
OverTime = typing.Annotated[Tensor, domain(TIME)]
PerSite = typing.Annotated[Tensor, domain(SITE)]
CurrentArray = typing.Annotated[Tensor, CURRENT]
Voltage = typing.Annotated[Quantity, VOLTAGE]
VoltageArray = typing.Annotated[Tensor, VOLTAGE]
Resistance = typing.Annotated[Quantity, RESISTANCE]


@pytest.fixture(autouse=True)
def _use_scc(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(implementation_module, "CausalizationStructuralAnalyzer", CausalizationStructuralAnalyzer)


@pytest.fixture(params=(PYTHON_BACKEND, JAX_BACKEND), ids=("python", "jax"))
def backend(request: pytest.FixtureRequest) -> Backend:
    return request.param


@system
class Pair(metaclass=Interface):
    x: Tensor = field()
    y: Tensor = field()
    p: float | Tensor = field()
    q: float | Tensor = field()
    weights: Tensor = field()


@equation
def balance(x: Tensor, y: Tensor, p: Tensor) -> bool:
    return x + y == p


@equation
def aggregate(x: Tensor, y: Tensor, q: float) -> bool:
    return 0.5 * x.sum(TIME) - y == q


@pytest.mark.parametrize("reduction", ("sum", "mean"))
@pytest.mark.parametrize("output", ("x", "y"))
def test_reduction_infers_a_tensor_and_shared_scalar(backend: Backend, reduction: str, output: str) -> None:
    @equation
    def second(x: Tensor, y: Tensor, q: float) -> bool:
        return (0.5 * x.sum(TIME) if reduction == "sum" else x.mean(TIME)) - y == q

    with backend, Implementation(Pair, [balance, second]):
        result = getattr(Pair(p=tensor([2.0, 4.0, 6.0], over=TIME), q=1.0), output)
    shared = 2.0 if reduction == "sum" else 1.5
    if output == "x":
        assert isinstance(result, Tensor)
        assert result.domain == domain(TIME)
        np.testing.assert_allclose(result.values, np.array([2.0, 4.0, 6.0]) - shared, atol=1e-5)
    else:
        assert not isinstance(result, Tensor)
        assert float(result) == pytest.approx(shared, abs=1e-5)


@pytest.mark.parametrize("output", ("x", "y"))
def test_inferred_shared_value_can_have_a_subset_domain(backend: Backend, output: str) -> None:
    @equation
    def second(x: Tensor, y: Tensor, q: Tensor) -> bool:
        return 0.5 * x.sum(TIME) - y == q

    values = np.array([[2.0, 4.0, 6.0], [4.0, 8.0, 12.0]])
    with backend, Implementation(Pair, [balance, second]):
        result = getattr(Pair(p=tensor(values, over=(SITE, TIME)), q=tensor([1.0, 2.0], over=SITE)), output)
    assert isinstance(result, Tensor)
    if output == "x":
        assert result.domain == domain(SITE, TIME)
        np.testing.assert_allclose(result.values, values - np.array([2.0, 4.0])[:, None], atol=1e-5)
    else:
        assert result.domain == domain(SITE)
        np.testing.assert_allclose(result.values, [2.0, 4.0], atol=1e-5)


def _alternative_layouts(hint: str | None) -> tuple[BiltModelConstituent, ...]:
    if hint == "x":

        @equation
        def first(x: PerSite, y: Tensor, p: Tensor) -> bool:
            return x + y == p
    elif hint == "y":

        @equation
        def first(x: Tensor, y: OverTime, p: Tensor) -> bool:
            return x + y == p
    else:
        first = balance

    @equation
    def second(x: Tensor, y: Tensor, q: float) -> bool:
        return x.sum(SITE) + y.sum(TIME) == q

    return first, second


def test_incomparable_minimal_layouts_require_a_hint(backend: Backend) -> None:
    values = np.array([[2.0, 4.0, 6.0], [4.0, 8.0, 12.0]])
    with (
        backend,
        Implementation(Pair, _alternative_layouts(None)),
        pytest.raises(BiltExecutionPlanError, match="Ambiguous"),
    ):
        _ = Pair(p=tensor(values, over=(SITE, TIME)), q=1.0).x


@pytest.mark.parametrize("hint", ("x", "y"))
@pytest.mark.parametrize("output", ("x", "y"))
def test_one_hint_selects_between_two_valid_layouts(backend: Backend, hint: str, output: str) -> None:
    values = np.array([[2.0, 4.0, 6.0], [4.0, 8.0, 12.0]])
    with backend, Implementation(Pair, _alternative_layouts(hint)):
        result = getattr(Pair(p=tensor(values, over=(SITE, TIME)), q=1.0), output)
    if hint == "x":
        x = np.array([15.0, 19.0])
        y = values - x[:, None]
        expected_domain = domain(SITE) if output == "x" else domain(SITE, TIME)
    else:
        y = np.array([-14.0, -11.0, -8.0])
        x = values - y[None, :]
        expected_domain = domain(SITE, TIME) if output == "x" else domain(TIME)
    assert result.domain == expected_domain
    np.testing.assert_allclose(result.values, x if output == "x" else y, atol=1e-5)


def test_exact_hint_can_require_a_larger_representation(backend: Backend) -> None:
    @equation
    def first(x: Tensor, y: OverTime, p: Tensor) -> bool:
        return x + y == p

    with backend, Implementation(Pair, [first, aggregate]):
        result = Pair(p=tensor([2.0, 4.0, 6.0], over=TIME), q=1.0).y
    assert result.domain == domain(TIME)
    np.testing.assert_allclose(result.values, [2.0, 2.0, 2.0], atol=1e-5)


def test_hint_conflicting_with_a_required_axis_is_rejected(backend: Backend) -> None:
    @equation
    def first(x: Shared, y: Tensor, p: Tensor) -> bool:
        return x + y == p

    with (
        backend,
        Implementation(Pair, [first, aggregate]),
        pytest.raises(BiltExecutionPlanError, match="No consistent"),
    ):
        _ = Pair(p=tensor([2.0, 4.0], over=TIME), q=1.0).x


def test_reduction_cannot_invent_axes_absent_from_known_inputs(backend: Backend) -> None:
    with (
        backend,
        Implementation(Pair, [balance, aggregate]),
        pytest.raises(BiltExecutionPlanError, match="block inputs"),
    ):
        _ = Pair(p=2.0, q=1.0).x


@pytest.mark.parametrize("output", ("current", "bus_voltage"))
def test_circuit_infers_current_and_voltage_domains_from_unit_only_annotations(backend: Backend, output: str) -> None:
    @system
    class Circuit(metaclass=Interface):
        current: CurrentArray = field()
        bus_voltage: Voltage = field()
        source_voltage: VoltageArray = field()
        branch_resistance: Resistance = field()
        load_resistance: Resistance = field()

    @equation
    def branch_balance(
        current: CurrentArray, bus_voltage: Voltage, source_voltage: VoltageArray, branch_resistance: Resistance
    ) -> bool:
        return branch_resistance * current + bus_voltage == source_voltage

    @equation
    def load_balance(current: CurrentArray, bus_voltage: Voltage, load_resistance: Resistance) -> bool:
        return bus_voltage == load_resistance * current.sum(BRANCH)

    with backend, Implementation(Circuit, [branch_balance, load_balance]):
        result = getattr(
            Circuit(
                source_voltage=tensor([10.0, 12.0, 14.0], over=BRANCH, unit=volt),
                branch_resistance=1.0 * ohm,
                load_resistance=0.5 * ohm,
            ),
            output,
        )
    if output == "current":
        assert isinstance(result, Tensor)
        assert result.domain == domain(BRANCH)
        assert result.unit == ampere
        np.testing.assert_allclose(result.values, [2.8, 4.8, 6.8], atol=1e-5)
    else:
        assert isinstance(result, Quantity)
        assert result.unit == volt
        assert float(result.value) == pytest.approx(7.2, abs=1e-5)


@pytest.mark.parametrize("output", ("x", "y"))
def test_chained_reductions_infer_multiple_axes(backend: Backend, output: str) -> None:
    @equation
    def second(x: Tensor, y: Tensor, q: float) -> bool:
        return x.sum(BRANCH).mean(TIME) - y == q

    values = np.array([[2.0, 4.0, 6.0], [4.0, 8.0, 12.0]])
    with backend, Implementation(Pair, [balance, second]):
        result = getattr(Pair(p=tensor(values, over=(TIME, BRANCH)), q=2.0), output)
    if output == "x":
        assert result.domain == domain(TIME, BRANCH)
        np.testing.assert_allclose(result.values, values - 4.0, atol=1e-5)
    else:
        assert float(result) == pytest.approx(4.0, abs=1e-5)


@pytest.mark.parametrize("output", ("x", "y"))
def test_coordinate_selection_can_infer_a_shared_scalar(backend: Backend, output: str) -> None:
    @equation
    def second(x: Tensor, y: Tensor, q: float) -> bool:
        return x - 0.25 * y.sel(TIME, "reference") == q

    with EvaluationContext() as evaluation, backend, Implementation(Pair, [balance, second]):
        evaluation.domains.bind(coordinate_axis(TIME, ("first", "reference", "last")))
        result = getattr(Pair(p=tensor([2.0, 4.0, 6.0], over=TIME), q=0.0), output)
    if output == "x":
        assert not isinstance(result, Tensor)
        assert float(result) == pytest.approx(0.8, abs=1e-5)
    else:
        assert result.domain == domain(TIME)
        np.testing.assert_allclose(result.values, [1.2, 3.2, 5.2], atol=1e-5)


def test_inferred_layout_stays_valid_when_coordinate_counts_change(backend: Backend) -> None:
    with backend, Implementation(Pair, [balance, aggregate]):
        first = Pair(p=tensor([2.0, 4.0, 6.0], over=TIME), q=1.0).x
        second = Pair(p=tensor([2.0, 4.0], over=TIME), q=1.0).x
    np.testing.assert_allclose(first.values, [0.0, 2.0, 4.0], atol=1e-5)
    np.testing.assert_allclose(second.values, [1.0, 3.0], atol=1e-5)

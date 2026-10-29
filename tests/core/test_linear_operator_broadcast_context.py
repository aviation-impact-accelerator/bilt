import dataclasses
import typing

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bilt import Domain, Implementation, Interface, Quantity, Tensor, dim, domain, equation, field, system, tensor
from bilt._backend._jax._backend import JAX_BACKEND
from bilt._backend._python._backend import PYTHON_BACKEND
from bilt.core._evaluation_context import EvaluationContext
from bilt.core._evaluation_request import EvaluationRequest
from bilt.core._structural_analysis._causalization_block_solver._backend import CausalizationStructuralAnalyzer
from bilt.core._structural_analysis._causalization_block_solver._errors import BiltExecutionPlanError
from bilt.core._system import _implementation as implementation_module
from bilt.symbolic._dimensions import coordinate_axis
from bilt.units import fraction
from bilt.units.dimensions import DIMENSIONLESS, LENGTH
from bilt.units.units import kilometer, meter

if typing.TYPE_CHECKING:
    from bilt._backend import Backend
    from bilt.core._model_constituent import BiltModelConstituent

TIME = dim("time")
SITE = dim("site")
Shared = typing.Annotated[Tensor, Domain.scalar()]
OverTime = typing.Annotated[Tensor, domain(TIME)]
PerSite = typing.Annotated[Tensor, domain(SITE)]
LengthArray = typing.Annotated[Tensor, LENGTH]
SharedLength = typing.Annotated[Quantity, LENGTH, Domain.scalar()]
LengthScalar = typing.Annotated[Quantity, LENGTH]
RatioArray = typing.Annotated[Tensor, DIMENSIONLESS]


@pytest.fixture(autouse=True)
def _use_scc(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(implementation_module, "CausalizationStructuralAnalyzer", CausalizationStructuralAnalyzer)


@pytest.fixture(params=(PYTHON_BACKEND, JAX_BACKEND), ids=("python", "jax"))
def backend(request: pytest.FixtureRequest) -> Backend:
    return request.param


@system
class Pair(metaclass=Interface):
    x: float | Tensor = field()
    y: float | Tensor = field()
    p: float | Tensor = field()
    q: float | Tensor = field()
    weights: Tensor = field()
    offset: float | Tensor = field()


def _balance_for_layout(layout: str) -> BiltModelConstituent:
    if layout == "shared-x":

        @equation
        def first(x: Shared, y: OverTime, p: Tensor) -> bool:
            return x + y == p
    elif layout == "shared-y":

        @equation
        def first(x: OverTime, y: Shared, p: Tensor) -> bool:
            return x + y == p
    elif layout == "infer-x":

        @equation
        def first(x: Shared, y: Tensor, p: Tensor) -> bool:
            return x + y == p
    elif layout == "infer-y":

        @equation
        def first(x: Tensor, y: Shared, p: Tensor) -> bool:
            return x + y == p
    else:

        @equation
        def first(x: Tensor, y: Tensor, p: Tensor) -> bool:
            return x + y == p

    return first


def _weighted_operations(layout: str, reduction: str = "sum") -> tuple[BiltModelConstituent, ...]:
    first = _balance_for_layout(layout)

    @equation
    def second(x: Tensor, y: Tensor, weights: Tensor, q: float) -> bool:
        return getattr(weights * x + y, reduction)(TIME) == q

    return first, second


@pytest.mark.parametrize("layout", ("shared-x", "shared-y", "infer-x", "infer-y"))
@pytest.mark.parametrize("reduction", ("sum", "mean"))
@pytest.mark.parametrize("output", ("x", "y"))
def test_reductions_broadcast_shared_unknowns_before_reducing(
    backend: Backend, layout: str, reduction: str, output: str
) -> None:
    with backend, Implementation(Pair, _weighted_operations(layout, reduction)):
        result = getattr(
            Pair(
                p=tensor([2.0, 4.0], over=TIME),
                weights=tensor([1.0, 2.0], over=TIME),
                q=8.0 if reduction == "sum" else 4.0,
            ),
            output,
        )
    shared = layout.endswith(output)
    if shared:
        assert not isinstance(result, Tensor)
        assert float(result) == pytest.approx(2.0, abs=1e-5)
    else:
        assert isinstance(result, Tensor)
        assert result.domain == domain(TIME)
        np.testing.assert_allclose(result.values, [0, 2], atol=1e-5)


@pytest.mark.parametrize("reduction", ("sum", "mean"))
def test_inference_reports_the_newly_supported_alternative_layout(backend: Backend, reduction: str) -> None:
    with (
        backend,
        Implementation(Pair, _weighted_operations("unhinted", reduction)),
        pytest.raises(BiltExecutionPlanError, match="Ambiguous"),
    ):
        _ = Pair(p=tensor([2.0, 4.0], over=TIME), weights=tensor([1.0, 2.0], over=TIME), q=8.0).x


def _offset_operations(hinted: bool, reduction: str = "sum") -> tuple[BiltModelConstituent, ...]:
    if hinted:

        @equation
        def first(x: OverTime, y: Shared, p: Tensor) -> bool:
            return x + y == p
    else:

        @equation
        def first(x: Tensor, y: Tensor, p: Tensor) -> bool:
            return x + y == p

    @equation
    def second(x: Tensor, y: Tensor, offset: float, q: float) -> bool:
        return getattr(x + offset, reduction)(TIME) - y == q

    return first, second


@pytest.mark.parametrize("hinted", (False, True))
@pytest.mark.parametrize("reduction", ("sum", "mean"))
@pytest.mark.parametrize("output", ("x", "y"))
def test_scalar_constant_is_broadcast_inside_reduction(
    backend: Backend, hinted: bool, reduction: str, output: str
) -> None:
    with backend, Implementation(Pair, _offset_operations(hinted, reduction)):
        result = getattr(
            Pair(p=tensor([2.0, 4.0, 6.0], over=TIME), offset=2.0, q=10.0 if reduction == "sum" else 2.0), output
        )
    if output == "x":
        assert result.domain == domain(TIME)
        np.testing.assert_allclose(result.values, [0, 2, 4], atol=1e-5)
    else:
        assert not isinstance(result, Tensor)
        assert float(result) == pytest.approx(2.0, abs=1e-5)


@pytest.mark.parametrize("output", ("x", "y"))
def test_all_unknowns_can_be_scalar_when_known_values_supply_the_receiver_axis(backend: Backend, output: str) -> None:
    with backend, Implementation(Pair, _weighted_operations("unhinted")):
        result = getattr(Pair(p=6.0, weights=tensor([1.0, 2.0, 3.0], over=TIME), q=24.0), output)
    assert not isinstance(result, Tensor)
    assert float(result) == pytest.approx(2.0 if output == "x" else 4.0, abs=1e-5)


def test_constant_tensor_supplies_the_axis_for_a_scalar_unknown(backend: Backend) -> None:
    @equation
    def first(x: Tensor, y: Tensor, p: float) -> bool:
        return x + y == p

    @equation
    def second(x: Tensor, y: Tensor, offset: Tensor, q: float) -> bool:
        return (x + offset).sum(TIME) + y == q

    with backend, Implementation(Pair, [first, second]):
        result = Pair(p=6.0, offset=tensor([1.0, 2.0, 3.0], over=TIME), q=16.0).x
    assert not isinstance(result, Tensor)
    assert float(result) == pytest.approx(2.0, abs=1e-5)


@pytest.mark.parametrize("selection", ("isel", "sel"))
@pytest.mark.parametrize("output", ("x", "y"))
def test_scalar_offset_is_broadcast_before_selection(backend: Backend, selection: str, output: str) -> None:
    @equation
    def first(x: Tensor, y: Tensor, p: Tensor) -> bool:
        return x + y == p

    @equation
    def second(x: Tensor, y: Tensor, offset: float, q: Tensor) -> bool:
        return x - 0.25 * getattr(y + offset, selection)(TIME, 1 if selection == "isel" else "reference") == q

    with EvaluationContext() as evaluation, backend, Implementation(Pair, [first, second]):
        evaluation.domains.bind(coordinate_axis(TIME, ("start", "reference")))
        result = getattr(Pair(p=tensor([2.0, 4.0], over=TIME), q=tensor([0.0, 1.0], over=TIME), offset=2.0), output)
    assert result.domain == domain(TIME)
    np.testing.assert_allclose(result.values, [1, 2], atol=1e-5)


@pytest.mark.parametrize("selection", ("isel", "sel"))
def test_known_tensor_supplies_the_axis_for_selecting_a_shared_scalar(backend: Backend, selection: str) -> None:
    @equation
    def first(x: Tensor, y: Tensor, p: Tensor) -> bool:
        return x + y == p

    @equation
    def second(x: Tensor, y: Tensor, offset: Tensor, q: float) -> bool:
        return (
            getattr(y + offset, selection)(TIME, 0 if selection == "isel" else "reference") + 0.5 * x.isel(TIME, 1) == q
        )

    with EvaluationContext() as evaluation, backend, Implementation(Pair, [first, second]):
        evaluation.domains.bind(coordinate_axis(TIME, ("reference", "end")))
        result = Pair(p=tensor([2.0, 4.0], over=TIME), offset=tensor([2.0, 3.0], over=TIME), q=4.5).y
    assert not isinstance(result, Tensor)
    assert float(result) == pytest.approx(1.0, abs=1e-5)


@pytest.mark.parametrize("output", ("x", "y"))
def test_broadcast_context_preserves_units(backend: Backend, output: str) -> None:
    @system
    class Lengths(metaclass=Interface):
        x: LengthArray = field()
        y: SharedLength = field()
        p: LengthArray = field()
        q: LengthScalar = field()
        weights: RatioArray = field()
        offset: LengthScalar = field()

    @equation
    def first(x: LengthArray, y: SharedLength, p: LengthArray) -> bool:
        return x + y == p

    @equation
    def second(x: LengthArray, y: SharedLength, weights: RatioArray, offset: LengthScalar, q: LengthScalar) -> bool:
        return (weights * x + y + offset).sum(TIME) == q

    with backend, Implementation(Lengths, [first, second]):
        result = getattr(
            Lengths(
                p=tensor([0.002, 0.004, 0.006], over=TIME, unit=kilometer),
                weights=tensor([1.0, 2.0, 3.0], over=TIME, unit=fraction),
                offset=1.0 * meter,
                q=0.025 * kilometer,
            ),
            output,
        )
    assert result.unit == meter
    if output == "x":
        assert result.domain == domain(TIME)
        np.testing.assert_allclose(result.values, [0, 2, 4], atol=1e-5)
    else:
        assert isinstance(result, Quantity)
        assert float(result.value) == pytest.approx(2.0, abs=1e-5)


def test_cached_operator_uses_current_coordinate_count(backend: Backend) -> None:
    with backend, Implementation(Pair, _weighted_operations("infer-y")):
        first = Pair(p=tensor([2.0, 4.0, 6.0], over=TIME), weights=tensor([1.0, 2.0, 3.0], over=TIME), q=22.0).x
        second = Pair(p=tensor([2.0, 4.0], over=TIME), weights=tensor([1.0, 2.0], over=TIME), q=8.0).x
    assert isinstance(first, Tensor)
    assert isinstance(second, Tensor)
    np.testing.assert_allclose(first.values, [0, 2, 4], atol=1e-5)
    np.testing.assert_allclose(second.values, [0, 2], atol=1e-5)


@pytest.mark.parametrize("output", ("x", "y"))
def test_sum_context_handles_empty_tensor_unknowns(backend: Backend, output: str) -> None:
    with backend, Implementation(Pair, _offset_operations(hinted=True)):
        result = getattr(Pair(p=tensor([], over=TIME), offset=2.0, q=3.0), output)
    if output == "x":
        assert result.domain == domain(TIME)
        assert result.values.shape == (0,)
    else:
        assert float(result) == pytest.approx(-3.0, abs=1e-5)


def test_jax_derivatives_include_the_repeated_shared_value() -> None:
    with EvaluationContext() as evaluation:
        evaluation.domains.bind_all((tensor([2.0, 4.0, 6.0], over=TIME),))
        case = CausalizationStructuralAnalyzer(_weighted_operations("infer-y"), JAX_BACKEND).analyse(
            EvaluationRequest(
                output="x", input_domains={"p": domain(TIME), "weights": domain(TIME), "q": Domain.scalar()}
            )
        )
    compiled = JAX_BACKEND.chain_case(case)
    weights = jnp.array([1.0, 2.0, 3.0])
    derivative = jax.jacrev(lambda p: compiled.callable(p=p, weights=weights, q=22.0))(jnp.array([2.0, 4.0, 6.0]))
    np.testing.assert_allclose(derivative, np.eye(3) - np.array([1.0, 2.0, 3.0])[None, :] / 3, atol=1e-5)


def test_zero_intrinsic_is_required_only_for_operator_blocks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        PYTHON_BACKEND, "intrinsic_bindings", dataclasses.replace(PYTHON_BACKEND.intrinsic_bindings, linear_zero=None)
    )
    with (
        Implementation(Pair, _weighted_operations("infer-y")),
        pytest.raises(BiltExecutionPlanError, match="does not support joint"),
    ):
        _ = Pair(p=tensor([2.0, 4.0], over=TIME), weights=tensor([1.0, 2.0], over=TIME), q=8.0).x

    @equation
    def first(x: float, y: float, p: float) -> bool:
        return x + y == p

    @equation
    def second(x: float, y: float, q: float) -> bool:
        return x - y == q

    with Implementation(Pair, [first, second]):
        result = Pair(p=10.0, q=2.0).x
    assert not isinstance(result, Tensor)
    assert float(result) == pytest.approx(6.0)


@pytest.mark.parametrize("reordered", (False, True))
@pytest.mark.parametrize("output", ("x", "y"))
def test_shared_subdomain_broadcasts_over_the_reduced_axis(backend: Backend, reordered: bool, output: str) -> None:
    @equation
    def first(x: Tensor, y: PerSite, p: Tensor) -> bool:
        return x + y == p

    @equation
    def second(x: Tensor, y: PerSite, weights: Tensor, q: Tensor) -> bool:
        return (weights * x + y).sum(TIME) == q

    values = np.array([[2.0, 4.0, 6.0], [4.0, 8.0, 12.0]])
    over = (TIME, SITE) if reordered else (SITE, TIME)
    with backend, Implementation(Pair, [first, second]):
        result = getattr(
            Pair(
                p=tensor(values.T if reordered else values, over=over),
                weights=tensor([1.0, 2.0, 3.0], over=TIME),
                q=tensor([22.0, 44.0], over=SITE),
            ),
            output,
        )
    assert isinstance(result, Tensor)
    if output == "x":
        expected = values - np.array([2.0, 4.0])[:, None]
        assert result.domain == domain(*over)
        np.testing.assert_allclose(result.values, expected.T if reordered else expected, atol=1e-5)
    else:
        assert result.domain == domain(SITE)
        np.testing.assert_allclose(result.values, [2.0, 4.0], atol=1e-5)


def test_nested_reductions_keep_the_broadcast_context(backend: Backend) -> None:
    @equation
    def first(x: Tensor, y: Shared, p: Tensor) -> bool:
        return x + y == p

    @equation
    def second(x: Tensor, y: Shared, weights: Tensor, offset: float, q: float) -> bool:
        return (weights * x + y + offset).sum(TIME).mean(SITE) == q

    with backend, Implementation(Pair, [first, second]):
        result = Pair(
            p=tensor([[2.0, 4.0, 6.0], [4.0, 8.0, 12.0]], over=(SITE, TIME)),
            weights=tensor([1.0, 2.0, 3.0], over=TIME),
            offset=1.0,
            q=39.0,
        ).y
    assert not isinstance(result, Tensor)
    assert float(result) == pytest.approx(2.0, abs=1e-5)


def test_large_constant_does_not_erase_linear_coefficients(monkeypatch: pytest.MonkeyPatch) -> None:
    @equation
    def first(x: Tensor, y: Shared, p: Tensor) -> bool:
        return x + y == p

    @equation
    def second(x: Tensor, y: Shared, weights: Tensor, offset: float, q: float) -> bool:
        return (weights * x + y + offset).sum(TIME) == q

    original = np.linalg.solve
    matrices = []

    def solve(matrix: np.ndarray, rhs: np.ndarray) -> np.ndarray:
        matrices.append(matrix.copy())
        return original(matrix, rhs)

    monkeypatch.setattr(np.linalg, "solve", solve)
    with Implementation(Pair, [first, second]):
        result = Pair(p=tensor([0.0, 0.0], over=TIME), weights=tensor([1.0, 2.0], over=TIME), offset=1e20, q=2e20).x
    assert isinstance(result, Tensor)
    np.testing.assert_allclose(result.values, [0.0, 0.0], atol=1e-5)
    assert len(matrices) == 1
    assert np.linalg.matrix_rank(matrices[0]) == matrices[0].shape[-1]

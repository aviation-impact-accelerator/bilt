import typing

import numpy as np
import pytest

from bilt import Domain, Implementation, Interface, Quantity, Tensor, dim, domain, equation, field, system, tensor
from bilt._backend._jax._backend import JAX_BACKEND
from bilt._backend._python._backend import PYTHON_BACKEND
from bilt.core._structural_analysis._causalization_block_solver._backend import CausalizationStructuralAnalyzer
from bilt.core._structural_analysis._causalization_block_solver._errors import BiltExecutionPlanError
from bilt.core._system import _implementation as implementation_module
from bilt.errors import BiltEquationDimensionalityError
from bilt.units import fraction
from bilt.units.dimensions import DIMENSIONLESS, LENGTH, MASS
from bilt.units.units import kilometer, meter

if typing.TYPE_CHECKING:
    from bilt._backend import Backend

TIME = dim("time")
BRANCH = dim("branch")
LengthArray = typing.Annotated[Tensor, LENGTH]
LengthScalar = typing.Annotated[Quantity, LENGTH]
MassScalar = typing.Annotated[Quantity, MASS]
RatioArray = typing.Annotated[Tensor, DIMENSIONLESS]
OverTime = typing.Annotated[Tensor, domain(TIME)]
Grid = typing.Annotated[Tensor, domain(TIME, BRANCH)]
Shared = typing.Annotated[float, Domain.scalar()]


@pytest.fixture(params=(PYTHON_BACKEND, JAX_BACKEND), ids=("python", "jax"))
def backend(request: pytest.FixtureRequest) -> Backend:
    return request.param


@system
class Pair(metaclass=Interface):
    x: Tensor = field()
    y: float = field()
    p: Tensor = field()
    q: float = field()
    weights: Tensor = field()


def test_weighted_sum_broadcasts_named_dimensions(backend: Backend, analyser_factory: object) -> None:
    _ = analyser_factory

    @system
    class Model(metaclass=Interface):
        value: Tensor = field()
        weights: Tensor = field()
        answer: Tensor = field()

    @equation
    def weighted(answer: Tensor, value: Tensor, weights: Tensor) -> bool:
        return answer == (weights * value).sum(BRANCH)

    values = np.arange(1.0, 7.0).reshape(2, 3)
    with backend, Implementation(Model, [weighted]):
        result = Model(value=tensor(values, over=(TIME, BRANCH)), weights=tensor([1.0, 2.0, 3.0], over=BRANCH)).answer
    assert result.domain == domain(TIME)
    np.testing.assert_allclose(result.values, [14, 32], rtol=1e-5)


def test_composed_reductions_preserve_coherent_units(backend: Backend, analyser_factory: object) -> None:
    _ = analyser_factory

    @system
    class Lengths(metaclass=Interface):
        value: LengthArray = field()
        weights: RatioArray = field()
        answer: LengthScalar = field()

    @equation
    def weighted(answer: LengthScalar, value: LengthArray, weights: RatioArray) -> bool:
        return answer == (weights * value).sum(BRANCH).mean(TIME)

    with backend, Implementation(Lengths, [weighted]):
        result = Lengths(
            value=tensor(np.arange(1.0, 7.0).reshape(2, 3) / 1000, over=(TIME, BRANCH), unit=kilometer),
            weights=tensor([1.0, 2.0, 3.0], over=BRANCH, unit=fraction),
        ).answer
    assert isinstance(result, Quantity)
    assert result.unit == meter
    assert float(result.value) == pytest.approx(23.0, abs=1e-5)


def test_composed_reductions_still_reject_inconsistent_units() -> None:
    with pytest.raises(BiltEquationDimensionalityError, match="relates"):

        @equation
        def weighted(answer: MassScalar, value: LengthArray, weights: RatioArray) -> bool:
            return answer == (weights * value).sum(BRANCH).mean(TIME)


@pytest.mark.parametrize("output", ("x", "y"))
def test_weighted_reduction_of_unknowns_uses_joint_solve(
    backend: Backend, monkeypatch: pytest.MonkeyPatch, output: str
) -> None:
    monkeypatch.setattr(implementation_module, "ExhaustiveStructuralAnalyser", CausalizationStructuralAnalyzer)

    @equation
    def balance(x: OverTime, y: Shared, p: Tensor) -> bool:
        return x + y == p

    @equation
    def aggregate(x: OverTime, y: Shared, weights: Tensor, q: float) -> bool:
        return (weights * x).sum(TIME) - y == q

    with backend, Implementation(Pair, [balance, aggregate]):
        result = getattr(
            Pair(p=tensor([2.0, 4.0, 6.0], over=TIME), weights=tensor([1.0, 2.0, 3.0], over=TIME), q=14.0), output
        )
    if output == "x":
        assert result.domain == domain(TIME)
        np.testing.assert_allclose(result.values, [0, 2, 4], atol=1e-5)
    else:
        assert float(result) == pytest.approx(2.0, abs=1e-5)


def test_chained_reductions_of_unknowns_use_joint_solve(backend: Backend, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(implementation_module, "ExhaustiveStructuralAnalyser", CausalizationStructuralAnalyzer)

    @equation
    def balance(x: Grid, y: Shared, p: Tensor) -> bool:
        return x + y == p

    @equation
    def aggregate(x: Grid, y: Shared, q: float) -> bool:
        return x.sum(BRANCH).mean(TIME) - y == q

    values = np.array([[2.0, 4.0, 6.0], [4.0, 8.0, 12.0]])
    with backend, Implementation(Pair, [balance, aggregate]):
        result = Pair(p=tensor(values, over=(TIME, BRANCH)), q=2.0).x
    assert result.domain == domain(TIME, BRANCH)
    np.testing.assert_allclose(result.values, values - 4, atol=1e-5)


def test_selection_of_composed_unknown_expression_keeps_constant_terms(
    backend: Backend, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(implementation_module, "ExhaustiveStructuralAnalyser", CausalizationStructuralAnalyzer)

    @system
    class Model(metaclass=Interface):
        x: Tensor = field()
        y: Tensor = field()
        p: Tensor = field()
        q: Tensor = field()
        offset: Tensor = field()

    @equation
    def balance(x: OverTime, y: OverTime, p: Tensor) -> bool:
        return x + y == p

    @equation
    def anchored(x: OverTime, y: OverTime, offset: Tensor, q: Tensor) -> bool:
        return x - 0.25 * (y + offset).isel(TIME, 0) == q

    with backend, Implementation(Model, [balance, anchored]):
        result = Model(
            p=tensor([2.0, 4.0], over=TIME), q=tensor([0.0, 1.0], over=TIME), offset=tensor([2.0, 2.0], over=TIME)
        ).x
    np.testing.assert_allclose(result.values, [0.8, 1.8], atol=1e-5)


def test_nonlinear_composition_does_not_become_a_linear_block(
    backend: Backend, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(implementation_module, "ExhaustiveStructuralAnalyser", CausalizationStructuralAnalyzer)

    @equation
    def balance(x: OverTime, y: Shared, p: Tensor) -> bool:
        return x + y == p

    @equation
    def aggregate(x: OverTime, y: Shared, q: float) -> bool:
        return (x * x).sum(TIME) - y == q

    with (
        backend,
        Implementation(Pair, [balance, aggregate]),
        pytest.raises(BiltExecutionPlanError, match="coupled blocks"),
    ):
        _ = Pair(p=tensor([2.0, 4.0], over=TIME), q=1.0).x

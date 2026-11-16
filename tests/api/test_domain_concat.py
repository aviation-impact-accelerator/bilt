"""End-to-end tests for named-domain tensor concatenation."""

from typing import TYPE_CHECKING, Annotated

import numpy as np
import pytest

from bilt import Implementation, Interface, QuantityTensor, Tensor, dim, equation, field, system, tensor
from bilt._backend._jax._backend import JAX_BACKEND
from bilt._backend._python._backend import PYTHON_BACKEND
from bilt.core._evaluation_context import EvaluationContext
from bilt.errors import BiltAnalysisError
from bilt.symbolic._dimensions import coordinate_axis
from bilt.symbolic._domains import domain
from bilt.units import gram, kilogram
from bilt.units.dimensions import MASS

if TYPE_CHECKING:
    from bilt._backend._base_backend import Backend


type MassTensor = Annotated[QuantityTensor, MASS]


@pytest.fixture(autouse=True)
def _select_analyser(analyser_factory: object) -> None:
    _ = analyser_factory


@pytest.mark.parametrize("backend", (PYTHON_BACKEND, JAX_BACKEND))
def test_concatenate_historical_and_projected_dates(backend: Backend) -> None:
    historical_date = dim("historical_date")
    modeling_date = dim("modeling_date")
    combined_date = dim("combined_date")

    @system
    class Example(metaclass=Interface):
        historical: Tensor = field()
        projected: Tensor = field()
        combined: Tensor = field()

    @equation
    def combine(historical: Tensor, projected: Tensor, combined: Tensor) -> bool:
        return combined == Tensor.concat(
            (historical, projected),
            along=(historical_date, modeling_date),
            into=combined_date,
        )

    implementation = Implementation(Example, [combine])
    with backend, implementation:
        result = Example(
            historical=tensor([1.0, 2.0, 3.0], over=historical_date),
            projected=tensor([4.0, 5.0], over=modeling_date),
        ).combined

    assert result.domain == domain(combined_date)
    np.testing.assert_allclose(result.values, [1, 2, 3, 4, 5])


@pytest.mark.parametrize("backend", (PYTHON_BACKEND, JAX_BACKEND))
def test_concat_keeps_other_dimensions_and_axis_alignment(backend: Backend) -> None:
    scenario = dim("scenario")
    historical_date = dim("historical_date")
    modeling_date = dim("modeling_date")
    combined_date = dim("combined_date")

    @system
    class Example(metaclass=Interface):
        historical: Tensor = field()
        projected: Tensor = field()
        combined: Tensor = field()

    @equation
    def combine(historical: Tensor, projected: Tensor, combined: Tensor) -> bool:
        return combined == Tensor.concat(
            (historical, projected),
            along=(historical_date, modeling_date),
            into=combined_date,
        )

    implementation = Implementation(Example, [combine])
    with backend, implementation:
        result = Example(
            historical=tensor([[1.0, 3.0], [2.0, 4.0]], over=(historical_date, scenario)),
            projected=tensor([[5.0], [6.0]], over=(scenario, modeling_date)),
        ).combined

    # Canonical order is derived from input binding order; surviving scenario
    # should retain its own axis, even when inputs have different axis orders.
    assert result.domain == domain(scenario, combined_date)
    np.testing.assert_allclose(result.values, [[1, 2, 5], [3, 4, 6]])


@pytest.mark.parametrize("backend", (PYTHON_BACKEND, JAX_BACKEND))
def test_concat_can_feed_another_equation(backend: Backend) -> None:
    past = dim("past")
    future = dim("future")
    whole = dim("whole")

    @system
    class Example(metaclass=Interface):
        historical: Tensor = field()
        projected: Tensor = field()
        combined: Tensor = field()
        total: float = field()

    @equation
    def combine(historical: Tensor, projected: Tensor, combined: Tensor) -> bool:
        return combined == Tensor.concat((historical, projected), along=(past, future), into=whole)

    @equation
    def reduce(combined: Tensor, total: float) -> bool:
        return total == combined.sum(whole)

    implementation = Implementation(Example, [combine, reduce])
    with backend, implementation:
        result = Example(
            historical=tensor([1.0, 2.0], over=past),
            projected=tensor([3.0, 4.0], over=future),
        ).total

    assert result == pytest.approx(10.0)


@pytest.mark.parametrize("backend", (PYTHON_BACKEND, JAX_BACKEND))
def test_concat_propagates_coordinate_labels(backend: Backend) -> None:
    past = dim("past")
    future = dim("future")
    whole = dim("whole")

    @system
    class Example(metaclass=Interface):
        historical: Tensor = field()
        projected: Tensor = field()
        selected: float = field()

    @equation
    def select(historical: Tensor, projected: Tensor, selected: float) -> bool:
        concatenated = Tensor.concat((historical, projected), along=(past, future), into=whole)
        return selected == concatenated.sel(whole, 2025)

    implementation = Implementation(Example, [select])
    with EvaluationContext() as evaluation:
        evaluation.domains.bind_axis(coordinate_axis(past, (2023, 2024)))
        evaluation.domains.bind_axis(coordinate_axis(future, (2025, 2026)))
        with backend, implementation:
            result = Example(
                historical=tensor([1.0, 2.0], over=past),
                projected=tensor([3.0, 4.0], over=future),
            ).selected
        assert evaluation.domains.axis(whole).coordinates == (2023, 2024, 2025, 2026)

    assert result == pytest.approx(3.0)


@pytest.mark.parametrize("backend", (PYTHON_BACKEND, JAX_BACKEND))
def test_concat_sizes_resolve_fresh_each_evaluation(backend: Backend) -> None:
    a, b, out = dim("historical"), dim("projected"), dim("combined")

    @system
    class Example(metaclass=Interface):
        before: Tensor = field()
        after: Tensor = field()
        combined: Tensor = field()

    @equation
    def combine(before: Tensor, after: Tensor, combined: Tensor) -> bool:
        return combined == Tensor.concat((before, after), along=(a, b), into=out)

    implementation = Implementation(Example, [combine])
    with backend, implementation:
        short = Example(before=tensor([1], over=a), after=tensor([2], over=b)).combined
        long = Example(before=tensor([1, 2], over=a), after=tensor([3, 4, 5], over=b)).combined

    np.testing.assert_array_equal(short.values, [1, 2])
    np.testing.assert_array_equal(long.values, [1, 2, 3, 4, 5])


@pytest.mark.parametrize("backend", (PYTHON_BACKEND, JAX_BACKEND))
def test_concat_converts_units_and_preserves_quantity_kind(backend: Backend) -> None:
    historical_date = dim("historical_mass_date")
    projected_date = dim("projected_mass_date")
    combined_date = dim("combined_mass_date")

    @system
    class Example(metaclass=Interface):
        historical: MassTensor = field()
        projected: MassTensor = field()
        combined: MassTensor = field()

    @equation
    def combine(historical: MassTensor, projected: MassTensor, combined: MassTensor) -> bool:
        return combined == Tensor.concat(
            (historical, projected),
            along=(historical_date, projected_date),
            into=combined_date,
        )

    implementation = Implementation(Example, [combine])
    with backend, implementation:
        result = Example(
            historical=tensor([1.0, 2.0], over=historical_date, unit=kilogram),
            projected=tensor([500.0], over=projected_date, unit=gram),
        ).combined

    assert result.domain == domain(combined_date)
    assert isinstance(result, QuantityTensor)
    assert result.unit == kilogram
    np.testing.assert_allclose(result.values, [1.0, 2.0, 0.5])


def test_concat_rejects_invalid_api_inputs() -> None:
    a, b, c = dim("a"), dim("b"), dim("c")

    with pytest.raises(ValueError, match="at least two"):
        Tensor.concat((tensor([1], over=a),), along=(a,), into=c)
    with pytest.raises(ValueError, match="destination distinct"):
        Tensor.concat((tensor([1], over=a), tensor([2], over=b)), along=(a, b), into=a)


@pytest.mark.parametrize("backend", (PYTHON_BACKEND, JAX_BACKEND))
def test_concat_rejects_mismatched_other_dimensions(backend: Backend) -> None:
    a, b, c, x = (dim(name) for name in ("a", "b", "c", "x"))

    @system
    class Example(metaclass=Interface):
        left: Tensor = field()
        right: Tensor = field()
        result: Tensor = field()

    @equation
    def combine(left: Tensor, right: Tensor, result: Tensor) -> bool:
        return result == Tensor.concat((left, right), along=(a, b), into=c)

    implementation = Implementation(Example, [combine])
    with backend, implementation, pytest.raises(BiltAnalysisError, match="non-concatenated dimensions"):
        _ = Example(left=tensor([[1, 2]], over=(x, a)), right=tensor([3, 4], over=b)).result


@pytest.mark.parametrize("backend", (PYTHON_BACKEND, JAX_BACKEND))
def test_concat_rejects_partial_coordinate_metadata(backend: Backend) -> None:
    a, b, c = dim("past"), dim("future"), dim("combined")

    @system
    class Example(metaclass=Interface):
        left: Tensor = field()
        right: Tensor = field()
        result: Tensor = field()

    @equation
    def combine(left: Tensor, right: Tensor, result: Tensor) -> bool:
        return result == Tensor.concat((left, right), along=(a, b), into=c)

    implementation = Implementation(Example, [combine])
    with EvaluationContext() as evaluation:
        evaluation.domains.bind_axis(coordinate_axis(a, (2023, 2024)))
        with backend, implementation, pytest.raises(BiltAnalysisError, match="labelled and unlabelled"):
            _ = Example(left=tensor([1, 2], over=a), right=tensor([3, 4], over=b)).result

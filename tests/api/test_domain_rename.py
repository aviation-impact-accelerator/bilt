"""Regression tests for forward symbolic renaming of tensor dimensions."""

from typing import TYPE_CHECKING, cast

import numpy as np
import pytest

from bilt import Implementation, Interface, Tensor, dim, equation, field, system, tensor
from bilt._backend._jax._backend import JAX_BACKEND
from bilt._backend._python._backend import PYTHON_BACKEND
from bilt.core._evaluation_context import EvaluationContext
from bilt.symbolic._dimensions import Dim, coordinate_axis
from bilt.symbolic._domains import domain
from bilt.symbolic._expressions import tensor_symbol_expr

if TYPE_CHECKING:
    from bilt._backend._base_backend import Backend


@pytest.fixture(autouse=True)
def _select_structural_analyser(analyser_factory: object) -> None:
    _ = analyser_factory


@pytest.mark.parametrize("backend", (PYTHON_BACKEND, JAX_BACKEND))
def test_rename_then_reduce(backend: Backend) -> None:
    original = dim("original")
    renamed = dim("renamed")

    @system
    class Example(metaclass=Interface):
        values: Tensor = field()
        relabeled: Tensor = field()
        total: float = field()

    @equation
    def rename(values: Tensor, relabeled: Tensor) -> bool:
        return relabeled == values.rename({original: renamed})

    @equation
    def reduce(relabeled: Tensor, total: float) -> bool:
        return total == relabeled.sum(renamed)

    implementation = Implementation(Example, [rename, reduce])
    with backend, implementation:
        values = tensor([2.0, 4.0, 8.0], over=original)
        result = Example(values=values).relabeled
        total = Example(values=values).total

    assert result.domain == domain(renamed)
    np.testing.assert_allclose(result.values, [2.0, 4.0, 8.0])
    assert total == pytest.approx(14.0)


@pytest.mark.parametrize("backend", (PYTHON_BACKEND, JAX_BACKEND))
def test_rename_moves_axis_to_canonical_position(backend: Backend) -> None:
    region = dim("region")
    time = dim("time")
    new_region = dim("new_region")

    @system
    class Example(metaclass=Interface):
        values: Tensor = field()
        renamed: Tensor = field()

    @equation
    def rename(values: Tensor, renamed: Tensor) -> bool:
        return renamed == values.rename({region: new_region})

    implementation = Implementation(Example, [rename])
    with backend, implementation:
        result = Example(values=tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], over=(region, time))).renamed

    assert result.domain == domain(time, new_region)
    np.testing.assert_allclose(result.values, [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]])


@pytest.mark.parametrize("backend", (PYTHON_BACKEND, JAX_BACKEND))
def test_rename_propagates_coordinates_to_selection(backend: Backend) -> None:
    time = dim("historical_time")
    renamed = dim("selected_time")

    @system
    class Example(metaclass=Interface):
        values: Tensor = field()
        selected: float = field()

    @equation
    def select(values: Tensor, selected: float) -> bool:
        return selected == values.rename({time: renamed}).sel(renamed, 2025)

    implementation = Implementation(Example, [select])
    with EvaluationContext() as context:
        context.domains.bind_axis(coordinate_axis(time, (2024, 2025, 2026)))
        with backend, implementation:
            result = Example(values=tensor([2.0, 4.0, 8.0], over=time)).selected
        assert context.domains.axis(renamed).coordinates == (2024, 2025, 2026)

    assert result == pytest.approx(4.0)


@pytest.mark.parametrize("backend", (PYTHON_BACKEND, JAX_BACKEND))
def test_rename_can_use_different_extents_between_evaluations(backend: Backend) -> None:
    time = dim("time")
    renamed = dim("renamed")

    @system
    class Example(metaclass=Interface):
        values: Tensor = field()
        result: Tensor = field()

    @equation
    def relabel(values: Tensor, result: Tensor) -> bool:
        return result == values.rename({time: renamed})

    implementation = Implementation(Example, [relabel])
    with backend, implementation:
        first = Example(values=tensor([1.0, 2.0], over=time)).result
        second = Example(values=tensor([1.0, 2.0, 3.0], over=time)).result

    assert first.shape == (2,)
    assert second.shape == (3,)
    assert second.domain == domain(renamed)


def test_symbolic_rename_validates_mapping() -> None:

    source, target = dim("source"), dim("target")
    x = tensor_symbol_expr("x")

    with pytest.raises(TypeError, match="Dim to Dim"):
        x.rename(cast("dict[Dim, Dim]", {"source": target}))
    with pytest.raises(ValueError, match="same destination"):
        x.rename({source: target, dim("another"): target})


@pytest.mark.parametrize("backend", (PYTHON_BACKEND, JAX_BACKEND))
def test_rename_rejects_missing_source_dimension(backend: Backend) -> None:
    a, missing, target = dim("a"), dim("missing"), dim("target")

    @system
    class Example(metaclass=Interface):
        x: Tensor = field()
        y: Tensor = field()

    @equation
    def rename(x: Tensor, y: Tensor) -> bool:
        return y == x.rename({missing: target})

    implementation = Implementation(Example, [rename])
    with backend, implementation, pytest.raises(Exception, match=r"rename|dimension"):
        _ = Example(x=tensor([1, 2], over=a)).y


@pytest.mark.parametrize("backend", (PYTHON_BACKEND, JAX_BACKEND))
def test_rename_rejects_duplicate_result_dimension(backend: Backend) -> None:
    a, b = dim("a"), dim("b")

    @system
    class Example(metaclass=Interface):
        x: Tensor = field()
        y: Tensor = field()

    @equation
    def rename(x: Tensor, y: Tensor) -> bool:
        return y == x.rename({a: b})

    implementation = Implementation(Example, [rename])
    with backend, implementation, pytest.raises(Exception, match=r"duplicate|dimension"):
        _ = Example(x=tensor([[1, 2], [3, 4]], over=(a, b))).y

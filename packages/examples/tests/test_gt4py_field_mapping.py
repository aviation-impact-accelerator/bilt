"""Axis-mapping tests that do not require GT4Py itself."""

import sys
import types

import numpy as np
import pytest

from bilt import Domain, dim, tensor
from bilt.core._evaluation_context import EvaluationContext
from examples.gt4py import CartesianFieldMapping


class FakeField:
    def __init__(self, dimensions: list[str], values: np.ndarray) -> None:
        self.dimensions = dimensions
        self.values = values

    def asnumpy(self) -> np.ndarray:
        return self.values


@pytest.fixture
def mapping(monkeypatch: pytest.MonkeyPatch) -> tuple[CartesianFieldMapping, object, object]:
    gt4py = types.ModuleType("gt4py")
    gt4py.__path__ = []  # type: ignore[attr-defined]
    next_module = types.ModuleType("gt4py.next")
    next_module.as_field = lambda dims, values: FakeField(dims, values)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "gt4py", gt4py)
    monkeypatch.setitem(sys.modules, "gt4py.next", next_module)
    x = dim("x")
    y = dim("y")
    return CartesianFieldMapping(((x, "I"), (y, "J"))), x, y


def test_tensor_axis_mapping_round_trip(mapping: tuple[CartesianFieldMapping, object, object]) -> None:
    adapter, x, y = mapping
    original = np.arange(12).reshape(3, 4)
    source = tensor(original, over=(y, x))

    field = adapter.to_field(source)

    assert field.dimensions == ["I", "J"]
    assert field.values.shape == (4, 3)
    np.testing.assert_array_equal(field.values, original.T)
    restored = adapter.to_tensor(field, domain=Domain((y, x)))
    np.testing.assert_array_equal(restored.values, original)
    assert restored.domain == source.domain


def test_runtime_array_uses_evaluation_order(mapping: tuple[CartesianFieldMapping, object, object]) -> None:
    adapter, x, y = mapping
    source = tensor(np.arange(12).reshape(3, 4), over=(y, x))

    with EvaluationContext() as evaluation:
        evaluation.domains.bind(source)
        field = adapter.to_runtime_field(source.values)
        restored = adapter.to_runtime_array(field)

    np.testing.assert_array_equal(field.values, source.values.T)
    np.testing.assert_array_equal(restored, source.values)


def test_raw_arrays_require_explicit_domain(mapping: tuple[CartesianFieldMapping, object, object]) -> None:
    adapter, _, _ = mapping
    with pytest.raises(ValueError, match="require domain"):
        adapter.to_field(np.zeros((2, 3)))


def test_extra_dimensions_are_rejected(mapping: tuple[CartesianFieldMapping, object, object]) -> None:
    adapter, x, y = mapping
    scenario = dim("scenario")
    source = tensor(np.zeros((2, 3, 4)), over=(x, y, scenario))
    with pytest.raises(ValueError, match="additional batch dimensions"):
        adapter.to_field(source)


def test_duplicate_mapping_is_rejected() -> None:
    x = dim("x")
    y = dim("y")
    with pytest.raises(ValueError, match="Bilt dimension"):
        CartesianFieldMapping(((x, "I"), (x, "J")))
    with pytest.raises(ValueError, match="GT4Py dimension"):
        CartesianFieldMapping(((x, "I"), (y, "I")))

"""Plain Python methods receive their original unit arguments after compilation."""

import typing
from types import SimpleNamespace

from bilt import PYTHON_BACKEND, Implementation, Interface, field, system, transform
from bilt.units import Unit, kilometer, meter

if typing.TYPE_CHECKING:
    import pytest


@system
class Extraction(metaclass=Interface):
    record: SimpleNamespace = field()
    extracted: float = field()


def test_unit_erasure_preserves_unit_arguments_to_plain_methods() -> None:
    expected = 3.0
    received: list[object] = []

    def convert_to(unit: object) -> SimpleNamespace:
        received.append(unit)
        return SimpleNamespace(value=expected)

    @transform
    def extracted(record: SimpleNamespace) -> float:
        local = record
        return local.convert_to(meter).value

    model = Implementation(Extraction, [extracted])
    inputs: dict[str, object] = {
        "record": SimpleNamespace(convert_to=convert_to),
    }

    result = model.evaluate(inputs, "extracted", PYTHON_BACKEND)

    assert len(received) == 1
    assert received[0] is meter
    assert result == expected


def _extraction_using(unit: Unit) -> Implementation:
    @transform
    def extracted(record: SimpleNamespace) -> float:
        local = record
        return local.convert_to(unit).value

    return Implementation(Extraction, [extracted])


def test_captured_units_are_local_to_each_compiled_transform() -> None:
    expected = 3.0
    received: list[object] = []

    def convert_to(unit: object) -> SimpleNamespace:
        received.append(unit)
        return SimpleNamespace(value=expected)

    inputs: dict[str, object] = {
        "record": SimpleNamespace(convert_to=convert_to),
    }
    first = _extraction_using(meter)
    second = _extraction_using(kilometer)

    first_result = first.evaluate(inputs, "extracted", PYTHON_BACKEND)
    second_result = second.evaluate(inputs, "extracted", PYTHON_BACKEND)
    repeated_result = first.evaluate(inputs, "extracted", PYTHON_BACKEND)

    expected_length = 3

    assert len(received) == expected_length
    assert received[0] is meter
    assert received[1] is kilometer
    assert received[2] is meter
    assert first_result == expected
    assert second_result == expected
    assert repeated_result == expected


def test_other_runtime_calls_still_use_the_backend_namespace(monkeypatch: pytest.MonkeyPatch) -> None:
    expected = 3.0

    def finish(value: float) -> float:
        msg = f"The captured helper must not replace the backend binding: {value}"
        raise AssertionError(msg)

    def runtime_finish(value: float) -> float:
        return value

    def convert_to(unit: object) -> SimpleNamespace:
        assert unit is meter
        return SimpleNamespace(value=expected)

    monkeypatch.setitem(PYTHON_BACKEND.compilation_namespace, "finish", runtime_finish)

    @transform
    def extracted(record: SimpleNamespace) -> float:
        local = record
        value = local.convert_to(meter).value
        return finish(value)

    model = Implementation(Extraction, [extracted])
    inputs: dict[str, object] = {
        "record": SimpleNamespace(convert_to=convert_to),
    }

    result = model.evaluate(inputs, "extracted", PYTHON_BACKEND)

    assert result == expected

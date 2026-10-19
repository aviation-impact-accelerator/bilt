"""A transform's body means in Bilt what it means in Python on quantities, whatever unit each quantity arrived in."""

import typing
from types import SimpleNamespace

import pytest

from bilt import JAX_BACKEND, PYTHON_BACKEND, Implementation, Interface, field, system, transform
from bilt.errors import BiltDefinitionError
from bilt.units import Quantity, meter, units
from bilt.units.dimensions import LENGTH
from bilt.units.units import absolute_celsius, kilometer, second

if typing.TYPE_CHECKING:
    from bilt._backend._base_backend import Backend
    from bilt.core._transform import BasicTransform

Length = typing.Annotated[Quantity, LENGTH]


@system
class Stretch(metaclass=Interface):
    length: Length = field()
    stretched: Length = field()
    span_km: float = field()


def _stretch(stretched: BasicTransform) -> Quantity:
    with Implementation(Stretch, [stretched]):
        return Stretch(length=1.0 * kilometer).stretched


@pytest.mark.parametrize("backend", (PYTHON_BACKEND, JAX_BACKEND))
def test_a_quantity_constant_means_what_it_says_whatever_unit_the_input_arrived_in(backend: Backend) -> None:
    @transform
    def stretched(length: Length) -> Length:
        return length + 1.0 * meter - 0.5 * meter

    with backend:
        assert _stretch(stretched) == 1000.5 * meter


def test_a_constructed_quantity_means_the_same_as_a_product() -> None:
    @transform
    def stretched(length: Length) -> Length:
        return length + Quantity(1.0, kilometer)

    assert _stretch(stretched) == 2000.0 * meter


def test_a_dimensionality_property_is_read() -> None:
    @transform
    def stretched(length: Length) -> Length:
        return length * 2.0 if not LENGTH.is_dimensionless else length

    assert _stretch(stretched) == 2000.0 * meter


def test_an_enclosing_quantity_is_honoured() -> None:
    offset = 1.0 * kilometer

    @transform
    def stretched(length: Length) -> Length:
        return length + offset

    assert _stretch(stretched) == 2000.0 * meter


def test_a_unit_reached_through_a_module_is_honoured() -> None:
    @transform
    def stretched(length: Length) -> Length:
        return length + 1.0 * units.kilometer

    assert _stretch(stretched) == 2000.0 * meter


def test_a_unit_is_a_scale_in_an_open_expression() -> None:
    @transform
    def stretched(length: Length) -> Length:
        return length / kilometer * meter

    assert _stretch(stretched) == 1.0 * meter


def test_a_unit_brings_a_plain_port_into_the_unit_system() -> None:
    @transform
    def stretched(span_km: float) -> Length:
        return span_km * kilometer

    with Implementation(Stretch, [stretched]):
        assert Stretch(span_km=2.0).stretched == 2000.0 * meter


def test_a_local_name_shadowing_a_unit_is_the_local() -> None:
    @transform
    def stretched(length: Length) -> Length:
        meter = 2.0
        return length * meter

    assert _stretch(stretched) == 2000.0 * meter


def test_a_comprehension_variable_shadowing_a_unit_is_the_unit_outside_it() -> None:
    @transform
    def stretched(length: Length) -> Length:
        squares = [meter * meter for meter in [1.0, 2.0]]
        return length + squares[1] * meter

    assert _stretch(stretched) == 1004.0 * meter


def test_a_lambda_parameter_shadowing_a_unit_is_the_unit_outside_it() -> None:
    @transform
    def stretched(length: Length) -> Length:
        longest = max([1.0, 2.0], key=lambda meter: meter * meter)
        return length + longest * meter

    assert _stretch(stretched) == 1002.0 * meter


def test_a_nested_function_parameter_shadowing_a_unit_is_the_unit_outside_it() -> None:
    @transform
    def stretched(length: Length) -> Length:
        def doubled(meter: float) -> float:
            return meter * 2.0

        return length + doubled(2.0) * meter

    assert _stretch(stretched) == 1004.0 * meter


def test_a_conversion_answers_the_same_quantity() -> None:
    @transform
    def stretched(length: Length) -> Length:
        return length.convert_to(kilometer)

    assert _stretch(stretched) == 1000.0 * meter


def test_a_converted_value_takes_a_declared_port_out_to_a_plain_one() -> None:
    @transform
    def span_km(length: Length) -> float:
        return length.convert_to(kilometer).value

    with Implementation(Stretch, [span_km]):
        assert Stretch(length=1000.0 * meter).span_km == 1.0


def test_a_constructed_quantity_brings_a_plain_port_into_the_unit_system() -> None:
    @transform
    def stretched(span_km: float) -> Length:
        return Quantity(span_km, kilometer)

    with Implementation(Stretch, [stretched]):
        assert Stretch(span_km=2.0).stretched == 2000.0 * meter


def test_a_constructed_quantity_takes_its_unit_by_keyword() -> None:
    @transform
    def stretched(span_km: float) -> Length:
        return Quantity(span_km, unit=kilometer)

    with Implementation(Stretch, [stretched]):
        assert Stretch(span_km=2.0).stretched == 2000.0 * meter


def test_a_constructed_quantity_takes_both_arguments_by_keyword() -> None:
    @transform
    def stretched(span_km: float) -> Length:
        return Quantity(value=span_km, unit=kilometer)

    with Implementation(Stretch, [stretched]):
        assert Stretch(span_km=2.0).stretched == 2000.0 * meter


def test_a_constructed_quantity_takes_its_keywords_in_either_order() -> None:
    @transform
    def stretched(span_km: float) -> Length:
        return Quantity(unit=kilometer, value=span_km)

    with Implementation(Stretch, [stretched]):
        assert Stretch(span_km=2.0).stretched == 2000.0 * meter


def test_a_conversion_in_one_transform_and_a_reading_in_the_next_agree_with_python() -> None:
    @transform
    def stretched(length: Length) -> Length:
        return length.convert_to(kilometer)

    @transform
    def span_km(stretched: Length) -> float:
        return stretched.convert_to(kilometer).value

    with Implementation(Stretch, [stretched, span_km]):
        assert Stretch(length=1.0 * kilometer).span_km == 1.0


def test_a_bare_value_is_refused_with_the_spelling_that_names_a_unit() -> None:
    with pytest.raises(
        BiltDefinitionError,
        match=r"`length\.value` depends on the unit `length` arrived in: write `length\.convert_to\(<unit>\)\.value`",
    ):

        @transform
        def stretched(length: Length) -> Length:
            return length * length.value


def test_a_bare_value_on_a_local_is_refused() -> None:
    with pytest.raises(BiltDefinitionError, match=r"`scaled\.value` depends on the unit `scaled` arrived in"):

        @transform
        def stretched(length: Length) -> Length:
            scaled = length * 2.0
            return scaled.value * meter


def test_a_bare_value_on_a_comprehension_variable_named_as_a_plain_parameter_is_refused() -> None:
    with pytest.raises(BiltDefinitionError, match=r"`span_km\.value` depends on the unit `span_km` arrived in"):

        @transform
        def stretched(length: Length, span_km: float) -> Length:
            spans = [span_km.value for span_km in [length, length]]
            return spans[0] * meter + span_km * kilometer


def test_a_unit_attribute_is_refused() -> None:
    with pytest.raises(BiltDefinitionError, match=r"`length\.unit` depends on the unit `length` arrived in"):

        @transform
        def stretched(length: Length) -> Length:
            return length * 2.0 if length.unit is meter else length


def test_an_absolute_unit_as_a_scale_is_refused() -> None:
    with pytest.raises(
        BiltDefinitionError,
        match=r"`absolute_celsius` is an absolute unit.*write `Quantity\(<value>, absolute_celsius\)`",
    ):

        @transform
        def stretched(length: Length) -> Length:
            return length * absolute_celsius


def test_converting_a_parameter_to_a_unit_of_another_dimensionality_is_refused() -> None:
    with pytest.raises(
        BiltDefinitionError,
        match=r"`length\.convert_to\(second\)` contradicts the signature: `length` is a Quantity of length.*"
        r"converts only to a relative unit such as meter",
    ):

        @transform
        def span_km(length: Length) -> float:
            return length.convert_to(second).value


def test_value_on_a_plain_parameter_fails_as_python_fails_it() -> None:
    @transform
    def stretched(span_km: float) -> Length:
        return span_km.value * kilometer  # pyrefly: ignore[missing-attribute]  # ty: ignore[unresolved-attribute]

    with (
        Implementation(Stretch, [stretched]),
        pytest.raises(AttributeError, match="'float' object has no attribute 'value'"),
    ):
        _ = Stretch(span_km=2.0).stretched


def test_unit_erasure_does_not_execute_runtime_calls() -> None:
    readings: list[float] = []

    def read_sensor() -> float:
        readings.append(1.0)
        return 1.0

    @transform
    def stretched(length: Length) -> Length:
        return length + read_sensor() * meter

    assert readings == []


def test_compiled_transform_executes_runtime_calls_on_each_evaluation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    readings: list[float] = []

    def read_sensor() -> float:
        reading = float(len(readings) + 1)
        readings.append(reading)
        return reading

    monkeypatch.setitem(
        PYTHON_BACKEND.compilation_namespace,
        "read_sensor",
        read_sensor,
    )

    @transform
    def stretched(length: Length) -> Length:
        return length + read_sensor() * meter

    assert readings == []

    model = Implementation(Stretch, [stretched])
    inputs: dict[str, object] = {"length": 1.0 * kilometer}

    first = model.evaluate(inputs, "stretched", PYTHON_BACKEND)
    second = model.evaluate(inputs, "stretched", PYTHON_BACKEND)

    assert readings == [1.0, 2.0]
    assert first == 1001.0 * meter
    assert second == 1002.0 * meter


def test_unit_erasure_preserves_value_on_a_plain_local_alias() -> None:
    @transform
    def extracted(record: SimpleNamespace) -> float:
        local = record
        return local.value

    result = extracted(SimpleNamespace(value=3.0))

    expected = 3.0
    result = extracted(SimpleNamespace(value=expected))

    assert result == expected


def test_unit_erasure_preserves_unit_arguments_to_plain_methods() -> None:
    expected = 3.0
    received: list[object] = []

    def convert_to(unit: object) -> SimpleNamespace:
        received.append(unit)
        return SimpleNamespace(value=expected)

    @system
    class Extraction(metaclass=Interface):
        record: SimpleNamespace = field()
        extracted: float = field()

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


def test_unit_erasure_preserves_value_on_a_constructed_plain_object() -> None:
    @transform
    def extracted(value: float) -> float:
        record = SimpleNamespace(value=value)
        return record.value

    expected = 3.0

    assert extracted(expected) == expected

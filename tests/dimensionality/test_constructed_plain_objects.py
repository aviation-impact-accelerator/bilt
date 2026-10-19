"""Only known plain constructions bypass Quantity attribute checks."""

from types import SimpleNamespace

import pytest

from bilt import transform
from bilt.errors import BiltDefinitionError
from bilt.units import Quantity, meter


def test_unit_erasure_preserves_value_on_a_constructed_plain_object() -> None:
    @transform
    def extracted(value: float) -> float:
        record = SimpleNamespace(value=value)
        return record.value

    expected = 3.0

    assert extracted(expected) == expected


def test_a_constructed_plain_object_can_be_aliased() -> None:
    @transform
    def extracted(value: float) -> float:
        record = SimpleNamespace(value=value)
        local = record
        return local.value

    expected = 3.0

    assert extracted(expected) == expected


def test_reassignment_does_not_mark_a_quantity_as_plain() -> None:
    with pytest.raises(BiltDefinitionError, match="record.value"):

        @transform
        def extracted(value: float) -> float:
            record: object = SimpleNamespace(value=value)
            record = Quantity(value, meter)
            return record.value


def test_a_shadowed_constructor_does_not_mark_a_quantity_as_plain() -> None:
    with pytest.raises(BiltDefinitionError, match="record.value"):

        @transform
        def extracted(value: float) -> float:
            namespace: type[SimpleNamespace] | type[Quantity] = SimpleNamespace
            namespace = Quantity
            record = namespace(value, meter)
            return record.value

"""Categorical values in ordinary Bilt tensors retain nominal enum identity."""

import enum

import numpy as np
import pytest

from bilt import Interface, Tensor, dim, field, system, tensor
from bilt.core._enum_encoding import enum_codes, enum_from_codes
from bilt.core._tensors._errors import BiltTensorDomainError
from bilt.errors import BiltEvaluationError
from bilt.units import year


class Mode(enum.Enum):
    MINIMUM = "minimum"
    AVERAGE = "average"
    MAXIMUM = "maximum"


class OtherMode(enum.Enum):
    MINIMUM = "minimum"


class Number(enum.IntEnum):
    FIRST = 11
    SECOND = 27


class Word(enum.StrEnum):
    FIRST = "first"
    SECOND = "second"


@system
class ScalarOnly(metaclass=Interface):
    mode: Mode = field()


@system
class ScalarOrTensor(metaclass=Interface):
    mode: Mode | Tensor = field()


@pytest.mark.parametrize(
    "members", ([Mode.MINIMUM, Mode.MAXIMUM], [Number.FIRST, Number.SECOND], [Word.FIRST, Word.SECOND])
)
def test_tensor_keeps_exact_enum_members(members: list[enum.Enum]) -> None:
    value = tensor(members, over=dim("scenario"))
    assert value.unit is None
    assert value.values.dtype == np.dtype(object)
    assert all(a is b for a, b in zip(value.values, members, strict=True))
    codes = enum_codes(value.values)
    assert np.array_equal(codes, [0, 2] if type(members[0]) is Mode else [0, 1])
    restored = enum_from_codes(type(members[0]), codes)
    assert all(a is b for a, b in zip(restored, members, strict=True))


def test_intenum_is_not_silently_converted_to_integers() -> None:
    value = tensor([Number.FIRST, Number.SECOND], over=dim("mode"))
    assert value.values.tolist() == [Number.FIRST, Number.SECOND]
    assert all(type(x) is Number for x in value.values)


def test_enum_port_accepts_scalar_or_matching_tensor() -> None:
    value = tensor([Mode.MINIMUM, Mode.MAXIMUM], over=dim("scenario"))
    ScalarOrTensor(mode=Mode.MINIMUM)
    ScalarOrTensor(mode=value)
    assert ScalarOrTensor.__bilt_enum_ports__["mode"] is Mode
    assert "mode" in ScalarOrTensor.__bilt_enum_tensor_ports__


def test_scalar_only_port_rejects_categorical_tensor() -> None:
    value = tensor([Mode.MINIMUM], over=dim("scenario"))
    with pytest.raises(BiltEvaluationError, match="requires a Mode enum member"):
        ScalarOnly(mode=value)


def test_categorical_port_rejects_numeric_and_other_enum_tensors() -> None:
    for value in (
        tensor([0, 1], over=dim("scenario")),
        tensor([OtherMode.MINIMUM], over=dim("scenario")),
        tensor([Number.FIRST], over=dim("scenario")),
        tensor([], over=dim("scenario")),
    ):
        with pytest.raises(BiltEvaluationError, match="requires a Tensor of Mode members"):
            ScalarOrTensor(mode=value)


def test_mixed_enum_classes_are_rejected_at_construction() -> None:
    with pytest.raises(BiltTensorDomainError, match="same enum class"):
        tensor([Mode.MINIMUM, OtherMode.MINIMUM], over=dim("scenario"))


def test_enum_tensor_does_not_accept_units() -> None:
    with pytest.raises(BiltTensorDomainError, match="cannot carry physical units"):
        tensor([Mode.MINIMUM], over=dim("scenario"), unit=year)

"""Typed enum leaves and deterministic backend ordinal conversion."""

import enum

import pytest

from bilt.core._enum_encoding import enum_code, enum_from_code
from bilt.symbolic._atoms import EnumType, IntType
from bilt.symbolic._expressions import as_tree
from bilt.symbolic._tree import enum_tree


class Mode(enum.Enum):
    OFF = "off"
    CRUISE = "cruise"


class Speed(enum.IntEnum):
    OFF = 11
    CRUISE = 27


class OtherSpeed(enum.IntEnum):
    OFF = 11
    CRUISE = 27


class Fuel(enum.StrEnum):
    OFF = "off"
    CRUISE = "cruise"


@pytest.mark.parametrize("member", (Mode.OFF, Speed.OFF, Fuel.OFF))
def test_enum_members_are_typed_symbolic_leaves(member: enum.Enum) -> None:
    tree = as_tree(member)
    assert tree is enum_tree(member)
    assert tree.head.atom_type is EnumType
    assert tree.head.value == (type(member), member.name)


def test_intenum_is_not_a_numeric_symbolic_literal() -> None:
    assert as_tree(Speed.OFF).head.atom_type is EnumType
    assert as_tree(11).head.atom_type is IntType
    assert as_tree(Speed.OFF) is not as_tree(11)


def test_different_enum_classes_never_share_a_symbolic_atom() -> None:
    assert as_tree(Speed.CRUISE) is not as_tree(OtherSpeed.CRUISE)
    assert as_tree(Speed.CRUISE).head.value == (Speed, "CRUISE")
    assert as_tree(OtherSpeed.CRUISE).head.value == (OtherSpeed, "CRUISE")


def test_aliases_canonicalise_to_one_member() -> None:
    class Aliased(enum.IntEnum):
        FIRST = 100
        SECOND = 100

    assert as_tree(Aliased.SECOND) is as_tree(Aliased.FIRST)
    assert as_tree(Aliased.SECOND).head.value == (Aliased, "FIRST")


@pytest.mark.parametrize("enum_type", (Mode, Speed, Fuel))
def test_ordinals_round_trip_without_using_python_values(enum_type: type[enum.Enum]) -> None:
    for ordinal, member in enumerate(enum_type):
        assert enum_code(member) == ordinal
        assert enum_from_code(enum_type, ordinal) is member


def test_decoding_requires_valid_nonnegative_ordinal() -> None:
    for invalid in (-1, 2):
        with pytest.raises(ValueError, match="outside the valid range"):
            enum_from_code(Mode, invalid)


def test_decoding_rejects_non_integral_code() -> None:
    with pytest.raises(TypeError):
        enum_from_code(Mode, 1.5)  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]

"""Prototype categorical enum match/case support in equations."""

import enum
lazy from contextlib import AbstractContextManager

import pytest

from bilt import JAX_BACKEND, PYTHON_BACKEND, Implementation, Interface, equation, field, system
from bilt.errors import BiltDefinitionError, BiltEvaluationError, BiltImplementationDimensionalityError
from bilt.symbolic._atoms import CONDITIONAL, EnumType
from bilt.symbolic._expressions import as_tree
from bilt.symbolic._walk import subtrees


class Mode(enum.Enum):
    OFF = "off"
    CRUISE = "cruise"
    BOOST = "boost"


class Speed(enum.IntEnum):
    OFF = 11
    CRUISE = 27
    BOOST = 53


class Fuel(enum.StrEnum):
    OFF = "idle"
    CRUISE = "cruise"
    BOOST = "boost"


@system
class EnumRig(metaclass=Interface):
    mode: Mode = field()
    power: float = field()
    output: float = field()


@equation
def enum_output(mode: Mode, power: float, output: float) -> bool:
    match mode:
        case Mode.OFF:
            selected = 0.0
        case Mode.CRUISE:
            selected = power * 0.5
        case Mode.BOOST:
            selected = power
        case _:
            selected = -1.0
    return output == selected


@system
class IntRig(metaclass=Interface):
    mode: Speed = field()
    power: float = field()
    output: float = field()


@equation
def int_output(mode: Speed, power: float, output: float) -> bool:
    match mode:
        case Speed.OFF:
            selected = 0.0
        case Speed.CRUISE:
            selected = power * 0.5
        case _:
            selected = power
    return output == selected


@system
class StringRig(metaclass=Interface):
    mode: Fuel = field()
    power: float = field()
    output: float = field()


@equation
def str_output(mode: Fuel, power: float, output: float) -> bool:
    match mode:
        case Fuel.OFF:
            selected = 0.0
        case Fuel.CRUISE:
            selected = power * 0.5
        case _:
            selected = power
    return output == selected


@pytest.mark.parametrize("backend", (PYTHON_BACKEND, JAX_BACKEND), ids=("python", "jax"))
@pytest.mark.parametrize(
    ("mode", "expected"),
    ((Mode.OFF, 0.0), (Mode.CRUISE, 5.0), (Mode.BOOST, 10.0)),
)
def test_enum_members_evaluate_with_both_backends(
    backend: AbstractContextManager[object], mode: Mode, expected: float
) -> None:
    implementation = Implementation(EnumRig, [enum_output])
    with backend, implementation:
        assert EnumRig(mode=mode, power=10.0).output == pytest.approx(expected)


@pytest.mark.parametrize("backend", (PYTHON_BACKEND, JAX_BACKEND), ids=("python", "jax"))
@pytest.mark.parametrize(
    ("mode", "expected"),
    ((Speed.OFF, 0.0), (Speed.CRUISE, 5.0), (Speed.BOOST, 10.0)),
)
def test_int_enum_uses_member_identity_not_underlying_integer(
    backend: AbstractContextManager[object], mode: Speed, expected: float
) -> None:
    implementation = Implementation(IntRig, [int_output])
    with backend, implementation:
        assert IntRig(mode=mode, power=10.0).output == pytest.approx(expected)


@pytest.mark.parametrize("backend", (PYTHON_BACKEND, JAX_BACKEND), ids=("python", "jax"))
@pytest.mark.parametrize(
    ("mode", "expected"),
    ((Fuel.OFF, 0.0), (Fuel.CRUISE, 5.0), (Fuel.BOOST, 10.0)),
)
def test_str_enum_is_encoded_to_numeric_backend_values(
    backend: AbstractContextManager[object], mode: Fuel, expected: float
) -> None:
    implementation = Implementation(StringRig, [str_output])
    with backend, implementation:
        assert StringRig(mode=mode, power=10.0).output == pytest.approx(expected)


def test_input_must_be_member_of_the_exact_enum_type() -> None:
    with pytest.raises(BiltEvaluationError, match="Mode enum member"):
        EnumRig(mode=Speed.CRUISE, power=10.0)  # pyrefly: ignore[bad-argument-type]  # ty: ignore[invalid-argument-type]
    with pytest.raises(BiltEvaluationError, match="Mode enum member"):
        EnumRig(mode="cruise", power=10.0)  # pyrefly: ignore[bad-argument-type]  # ty: ignore[invalid-argument-type]
    with pytest.raises(BiltEvaluationError, match="Speed enum member"):
        IntRig(mode=27, power=10.0)  # pyrefly: ignore[bad-argument-type]  # ty: ignore[invalid-argument-type]


def test_plain_fields_reject_enum_inputs() -> None:
    with pytest.raises(BiltEvaluationError, match="does not declare an enum"):
        EnumRig(mode=Mode.OFF, power=Speed.CRUISE)


def test_mismatched_enum_annotations_are_rejected() -> None:
    @equation
    def incompatible(mode: Fuel, power: float, output: float) -> bool:
        match mode:
            case Fuel.CRUISE:
                selected = power
            case _:
                selected = 0.0
        return output == selected

    with pytest.raises(BiltImplementationDimensionalityError, match="inconsistent enum types"):
        Implementation(EnumRig, [incompatible])


def test_enum_match_rejects_mixed_enum_classes() -> None:
    with pytest.raises(BiltDefinitionError, match="match subject's enum type"):

        @equation
        def bad_enum(mode: Mode, result: float) -> bool:
            match mode:
                case Fuel.CRUISE:
                    chosen = 1.0
                case _:
                    chosen = 0.0
            return result == chosen


def test_enum_match_rejects_numeric_literals() -> None:
    with pytest.raises(BiltDefinitionError, match="match subject's enum type"):

        @equation
        def bad_numeric(mode: Mode, result: float) -> bool:
            match mode:
                case 1:
                    chosen = 1.0
                case _:
                    chosen = 0.0
            return result == chosen


def test_enum_parameter_cannot_be_used_in_arithmetic() -> None:
    with pytest.raises(BiltDefinitionError, match="only be used as match subjects"):

        @equation
        def bad_arithmetic(mode: Mode, result: float) -> bool:
            return result == mode + 1  # pyrefly: ignore[unsupported-operation]  # ty: ignore[unsupported-operator]


def test_enum_port_cannot_be_solved_backwards() -> None:
    implementation = Implementation(EnumRig, [enum_output])
    with implementation, pytest.raises(BiltEvaluationError, match="forward-only match selectors"):
        _ = EnumRig(power=10.0, output=5.0).mode


def test_original_python_equation_retains_enum_matching() -> None:
    assert enum_output(Mode.CRUISE, 10.0, 5.0)
    assert not enum_output(Mode.CRUISE, 10.0, 10.0)


class OtherSpeed(enum.IntEnum):
    OFF = 11
    CRUISE = 27
    BOOST = 53


def test_symbolic_enum_members_retain_declared_class_and_member() -> None:
    # Enum identity survives tracing, rather than being flattened to an ordinal.
    leaves = [
        node for node in subtrees(int_output.__bilt_equation_tree__) if node.is_leaf and node.head.atom_type is EnumType
    ]
    assert {node.head.value for node in leaves} == {
        (Speed, "OFF"),
        (Speed, "CRUISE"),
    }


def test_numeric_and_enum_match_use_one_conditional_representation() -> None:
    assert any(node.head is CONDITIONAL for node in subtrees(int_output.__bilt_equation_tree__))
    assert any(node.head is CONDITIONAL for node in subtrees(enum_output.__bilt_equation_tree__))


def test_distinct_int_enum_types_do_not_alias_symbolic_constants() -> None:
    left = as_tree(Speed.CRUISE)
    right = as_tree(OtherSpeed.CRUISE)
    assert left is not right
    assert left.head.value == (Speed, "CRUISE")
    assert right.head.value == (OtherSpeed, "CRUISE")


def test_enum_member_is_not_a_numeric_symbolic_literal() -> None:
    assert as_tree(Speed.OFF).head.atom_type is EnumType
    assert as_tree(Speed.OFF) is not as_tree(11)


def test_enum_alias_canonicalises_to_original_member() -> None:
    class Aliased(enum.IntEnum):
        FIRST = 100
        SECOND = 100

    assert as_tree(Aliased.SECOND) is as_tree(Aliased.FIRST)
    assert as_tree(Aliased.SECOND).head.value == (Aliased, "FIRST")

"""Prototype categorical enum match/case support in equations."""

import enum
import functools
import typing
lazy from contextlib import AbstractContextManager

import jax
import jax.numpy as jnp
import pytest

from bilt import JAX_BACKEND, PYTHON_BACKEND, Implementation, Interface, equation, field, system
from bilt._backend._jax._backend import JAXBackend
from bilt.core._evaluation_context import EvaluationContext
from bilt.core._evaluation_request import EvaluationRequest
from bilt.errors import (
    BiltDefinitionError,
    BiltEvaluationError,
    BiltImplementationDimensionalityError,
)
from bilt.symbolic._atoms import CONDITIONAL, EnumType
from bilt.symbolic._domains import Domain
from bilt.symbolic._walk import subtrees

if typing.TYPE_CHECKING:
    from collections.abc import Callable
    from types import FunctionType

    from bilt.core._equation import BasicEquation




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
            scaled_power = power * 0.5
            selected = scaled_power
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
    with implementation, pytest.raises(BiltEvaluationError, match="forward-only"):
        _ = EnumRig(power=10.0, output=5.0).mode


def test_original_python_equation_retains_enum_matching() -> None:
    assert enum_output(Mode.CRUISE, 10.0, 5.0)
    assert not enum_output(Mode.CRUISE, 10.0, 10.0)


@pytest.mark.parametrize(
    ("interface", "relation", "modes"),
    (
        (EnumRig, enum_output, (Mode.OFF, Mode.CRUISE, Mode.BOOST)),
        (IntRig, int_output, (Speed.OFF, Speed.CRUISE, Speed.BOOST)),
        (StringRig, str_output, (Fuel.OFF, Fuel.CRUISE, Fuel.BOOST)),
    ),
    ids=("Enum", "IntEnum", "StrEnum"),
)
def test_jit_reuses_one_compilation_for_all_enum_members(
    monkeypatch: pytest.MonkeyPatch, interface: Interface, relation: BasicEquation, modes: tuple[enum.Enum, ...]
) -> None:
    backend = JAXBackend()
    traces: list[str] = []

    def count_jit_traces(function: FunctionType) -> Callable[..., object]:
        @functools.wraps(function)
        def traced(*args: object, **kwargs: object) -> object:
            traces.append(function.__name__)
            return function(*args, **kwargs)

        return jax.jit(traced)

    monkeypatch.setitem(backend.compilation_namespace, "jax.jit", count_jit_traces)
    implementation = Implementation(interface, [relation])
    request = EvaluationRequest("output", {"mode": Domain.scalar(), "power": Domain.scalar()})

    with EvaluationContext():
        compiled = implementation.get_evaluation(request, backend)
    assert hasattr(compiled.callable, "lower")

    assert implementation.evaluate({"mode": modes[0], "power": 10.0}, "output", backend) == pytest.approx(0.0)
    first_traces = traces.copy()
    assert first_traces.count(compiled.name) == 1

    for mode, expected in zip(modes[1:], (5.0, 10.0), strict=True):
        assert implementation.evaluate({"mode": mode, "power": 10.0}, "output", backend) == expected
        with EvaluationContext():
            assert implementation.get_evaluation(request, backend) is compiled

    assert traces == first_traces


@equation
def mixed_dtype_output(mode: Mode, output: float) -> bool:
    match mode:
        case Mode.OFF:
            selected = 0
        case _:
            selected = 2.5
    return output == selected


def test_jit_reports_incompatible_enum_branch_dtypes() -> None:
    implementation = Implementation(EnumRig, [mixed_dtype_output])
    with implementation, PYTHON_BACKEND:
        assert type(EnumRig(mode=Mode.OFF).output) is int
        assert type(EnumRig(mode=Mode.BOOST).output) is float

    with (
        implementation,
        JAX_BACKEND,
        pytest.raises(TypeError, match=r"(?s)cond branches must have equal output types.*dtypes do not match"),
    ):
        _ = EnumRig(mode=Mode.OFF).output


def test_jit_accepts_float_branches_with_weak_and_explicit_dtypes() -> None:
    implementation = Implementation(EnumRig, [enum_output])
    explicit_float = jnp.asarray(10.0, dtype=jnp.asarray(0.0).dtype)

    for mode, expected in ((Mode.OFF, 0.0), (Mode.CRUISE, 5.0)):
        result = implementation.evaluate({"mode": mode, "power": explicit_float}, "output", JAX_BACKEND)
        assert isinstance(result, jax.Array)
        assert result.dtype == explicit_float.dtype
        assert result == expected

def test_enum_match_uses_existing_conditional_tree() -> None:
    assert any(node.head is CONDITIONAL for node in subtrees(enum_output.__bilt_equation_tree__))
    members = [
        node for node in subtrees(int_output.__bilt_equation_tree__) if node.head.atom_type is EnumType
    ]
    assert {node.head.value for node in members} == {(Speed, "OFF"), (Speed, "CRUISE")}

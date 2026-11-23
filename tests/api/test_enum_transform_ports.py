"""Scalar categorical ports and directed transform execution with both backends."""

import enum
import typing

import pytest

from bilt import JAX_BACKEND, PYTHON_BACKEND, Implementation, Interface, field, system, transform
from bilt.errors import BiltEvaluationError, BiltImplementationDimensionalityError, BiltPortDeclarationError

if typing.TYPE_CHECKING:
    from bilt._backend._base_backend import Backend


class Mode(enum.Enum):
    OFF = "off"
    ON = "on"


class OtherMode(enum.Enum):
    OFF = "off"
    ON = "on"


class Speed(enum.IntEnum):
    OFF = 11
    ON = 27


class Fuel(enum.StrEnum):
    OFF = "off"
    ON = "on"


@system
class Selector(metaclass=Interface):
    mode: Mode = field()
    output: float = field()


@transform
def output(mode: Mode) -> float:
    return 5.0 if mode == Mode.ON else 0.0


@system
class ModeSource(metaclass=Interface):
    mode: Mode = field()


@transform
def mode() -> Mode:
    return Mode.ON


@transform
def speed() -> Speed:
    return Speed.ON


@system
class SpeedSource(metaclass=Interface):
    speed: Speed = field()


@transform
def fuel() -> Fuel:
    return Fuel.ON


@system
class FuelSource(metaclass=Interface):
    fuel: Fuel = field()


@transform
def echo_mode(mode: Mode) -> Mode:
    return mode


@system
class Echo(metaclass=Interface):
    mode: Mode = field()
    echo_mode: Mode = field()


@pytest.mark.parametrize("backend", (PYTHON_BACKEND, JAX_BACKEND), ids=("python", "jax"))
def test_enum_selector_transform_uses_encoded_input(backend: Backend) -> None:
    impl = Implementation(Selector, [output])
    expected_on = 5.0
    assert impl.evaluate({"mode": Mode.ON}, "output", backend) == expected_on
    assert impl.evaluate({"mode": Mode.OFF}, "output", backend) == 0.0


@pytest.mark.parametrize("backend", (PYTHON_BACKEND, JAX_BACKEND), ids=("python", "jax"))
def test_transform_can_produce_an_enum_member(backend: Backend) -> None:
    for interface, relation, expected in (
        (ModeSource, mode, Mode.ON),
        (SpeedSource, speed, Speed.ON),
        (FuelSource, fuel, Fuel.ON),
    ):
        implementation = Implementation(interface, [relation])
        assert implementation.evaluate({}, relation.__bilt_name__, backend) is expected


@pytest.mark.parametrize("backend", (PYTHON_BACKEND, JAX_BACKEND), ids=("python", "jax"))
def test_enum_pass_through_transform_returns_the_declared_type(backend: Backend) -> None:
    implementation = Implementation(Echo, [echo_mode])
    assert implementation.evaluate({"mode": Mode.ON}, "echo_mode", backend) is Mode.ON
    assert implementation.evaluate({"mode": Mode.OFF}, "echo_mode", backend) is Mode.OFF


def test_enum_values_are_validated_at_system_and_implementation_boundaries() -> None:
    with pytest.raises(BiltEvaluationError, match="requires a Mode enum member"):
        Selector(mode=OtherMode.ON)
    with pytest.raises(BiltEvaluationError, match="requires a Mode enum member"):
        Selector(mode="on")
    impl = Implementation(Selector, [output])
    with pytest.raises(BiltEvaluationError, match="requires a Mode enum member"):
        impl.evaluate({"mode": Speed.ON}, "output", PYTHON_BACKEND)


def test_shared_categorical_ports_require_identical_enum_classes() -> None:
    @transform
    def output(mode: OtherMode) -> float:
        return 1.0 if mode is OtherMode.ON else 0.0

    with pytest.raises(BiltImplementationDimensionalityError, match="inconsistent enum types"):
        Implementation(Selector, [output])


class Switches(enum.Flag):
    FIRST = enum.auto()


class Options(enum.IntFlag):
    FIRST = enum.auto()


@pytest.mark.parametrize("enum_type", (Switches, Options))
def test_unsupported_flag_ports_are_rejected(enum_type: type[enum.Flag]) -> None:
    with pytest.raises(BiltPortDeclarationError, match="Flag and IntFlag are unsupported"):

        @system
        class Bad(metaclass=Interface):
            flag: enum_type = field()  # ty: ignore[invalid-type-form]  # pyrefly: ignore[invalid-type-var]


def test_enum_unions_are_rejected() -> None:
    with pytest.raises(BiltPortDeclarationError, match="concrete enum class"):

        @system
        class Bad(metaclass=Interface):
            mode: Mode | None = field()


def test_enum_member_is_not_accepted_as_a_numeric_port() -> None:
    @system
    class Numeric(metaclass=Interface):
        value: float = field()

    with pytest.raises(BiltEvaluationError, match="does not declare an enum type"):
        Numeric(value=Mode.ON)

"""Compile exhaustive enum match transforms through Python and JAX backends."""

import enum
import typing

import pytest

from bilt import JAX_BACKEND, PYTHON_BACKEND, Implementation, Interface, field, system, transform
from bilt.errors import BiltCategoricalPortError

if typing.TYPE_CHECKING:
    from bilt._backend._base_backend import Backend


class Scenario(enum.Enum):
    MINIMUM = "min"
    AVERAGE = "average"
    MAXIMUM = "max"


class OtherScenario(enum.Enum):
    MINIMUM = "min"
    AVERAGE = "average"
    MAXIMUM = "max"


class Choice(enum.IntEnum):
    MINIMUM = 10
    AVERAGE = 20
    MAXIMUM = 30


class Fuel(enum.StrEnum):
    FOSSIL = "fossil"
    HYDROGEN = "hydrogen"


@system
class Prices(metaclass=Interface):
    scenario: Scenario = field()
    price: float = field()


@transform
def price(scenario: Scenario) -> float:
    match scenario:
        case Scenario.MINIMUM:
            return 0.7
        case Scenario.AVERAGE:
            return 1.0
        case Scenario.MAXIMUM:
            return 1.4
        case _:
            typing.assert_never(scenario)


@transform
def alternative_price(scenario: Scenario) -> float:
    match scenario:
        case Scenario.MINIMUM:
            result = 0.7
        case Scenario.AVERAGE:
            result = 1.0
        case Scenario.MAXIMUM:
            result = 1.4
    return result


@system
class AlternativePrices(metaclass=Interface):
    scenario: Scenario = field()
    alternative_price: float = field()


@system
class FuelChoice(metaclass=Interface):
    choice: Choice = field()
    fuel: Fuel = field()


@transform
def fuel(choice: Choice) -> Fuel:
    match choice:
        case Choice.MINIMUM:
            return Fuel.FOSSIL
        case Choice.AVERAGE:
            return Fuel.FOSSIL
        case Choice.MAXIMUM:
            return Fuel.HYDROGEN


@transform
def selected_price(scenario: Scenario) -> float:
    match scenario:
        case Scenario.MINIMUM:
            return 0.7
        case Scenario.AVERAGE:
            return 1.0
        case _:
            return 1.4


@system
class SelectedPrices(metaclass=Interface):
    scenario: Scenario = field()
    selected_price: float = field()


@pytest.mark.parametrize("backend", (PYTHON_BACKEND, JAX_BACKEND), ids=("python", "jax"))
def test_cyof_style_exhaustive_match_with_assert_never(backend: Backend) -> None:
    implementation = Implementation(Prices, [price])
    for mode, expected in ((Scenario.MINIMUM, 0.7), (Scenario.AVERAGE, 1.0), (Scenario.MAXIMUM, 1.4)):
        assert implementation.evaluate({"scenario": mode}, "price", backend) == pytest.approx(expected)
    assert price(Scenario.AVERAGE) == 1.0


@pytest.mark.parametrize("backend", (PYTHON_BACKEND, JAX_BACKEND), ids=("python", "jax"))
def test_enum_match_assigns_one_local_before_return(backend: Backend) -> None:
    implementation = Implementation(AlternativePrices, [alternative_price])
    assert implementation.evaluate({"scenario": Scenario.AVERAGE}, "alternative_price", backend) == 1.0
    assert implementation.evaluate({"scenario": Scenario.MAXIMUM}, "alternative_price", backend) == pytest.approx(1.4)


@pytest.mark.parametrize("backend", (PYTHON_BACKEND, JAX_BACKEND), ids=("python", "jax"))
def test_enum_match_can_produce_another_enum(backend: Backend) -> None:
    implementation = Implementation(FuelChoice, [fuel])
    assert implementation.evaluate({"choice": Choice.MINIMUM}, "fuel", backend) is Fuel.FOSSIL
    assert implementation.evaluate({"choice": Choice.MAXIMUM}, "fuel", backend) is Fuel.HYDROGEN


@pytest.mark.parametrize("backend", (PYTHON_BACKEND, JAX_BACKEND), ids=("python", "jax"))
def test_final_wildcard_uses_default_result(backend: Backend) -> None:
    implementation = Implementation(SelectedPrices, [selected_price])
    assert implementation.evaluate({"scenario": Scenario.MAXIMUM}, "selected_price", backend) == pytest.approx(1.4)


def test_wrong_enum_case_rejected_at_definition() -> None:
    with pytest.raises(BiltCategoricalPortError, match="exact selector enum class"):

        @transform
        def incorrect(scenario: Scenario) -> float:
            match scenario:
                case OtherScenario.MINIMUM:
                    return 0.7
                case _:
                    return 1.4


def test_duplicate_case_rejected() -> None:
    with pytest.raises(BiltCategoricalPortError, match="duplicate enum match case"):

        @transform
        def duplicate(scenario: Scenario) -> float:
            match scenario:
                case Scenario.MINIMUM:
                    return 0.7
                case Scenario.MINIMUM:
                    return 0.8
                case _:
                    return 1.4


def test_non_exhaustive_match_without_default_rejected() -> None:
    with pytest.raises(BiltCategoricalPortError, match="non-exhaustive"):

        @transform
        def incomplete(scenario: Scenario) -> float:  # pyrefly: ignore[bad-return]
            match scenario:
                case Scenario.MINIMUM:
                    return 0.7


def test_assert_never_requires_exhaustive_coverage() -> None:
    with pytest.raises(BiltCategoricalPortError, match="assert_never requires"):

        @transform
        def incomplete(scenario: Scenario) -> float:
            match scenario:
                case Scenario.MINIMUM:
                    return 0.7
                case _:
                    typing.assert_never(scenario)  # pyrefly: ignore[bad-argument-type]  # ty: ignore[invalid-argument-type]


def test_guards_rejected() -> None:
    with pytest.raises(BiltCategoricalPortError, match="guards are not supported"):

        @transform
        def guarded(scenario: Scenario) -> float:
            match scenario:
                case Scenario.MINIMUM if True:
                    return 0.7
                case _:
                    return 1.4


def test_different_enum_result_type_rejected() -> None:
    with pytest.raises(BiltCategoricalPortError, match="different enum class"):

        @transform
        def wrong(choice: Choice) -> Fuel:
            match choice:
                case Choice.MINIMUM:
                    return OtherScenario.MINIMUM  # pyrefly: ignore[bad-return]  # ty: ignore[invalid-return-type]
                case _:
                    return Fuel.FOSSIL


def test_direct_python_transform_is_not_modified() -> None:
    assert price(Scenario.MINIMUM) == pytest.approx(0.7)
    with pytest.raises(AssertionError):
        price(typing.cast("Scenario", OtherScenario.MINIMUM))

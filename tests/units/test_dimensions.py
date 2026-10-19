import pytest

from bilt.units import Dimension, Metadata
from bilt.units.dimensions import AMOUNT, DIMENSIONLESS, MASS, TIME


@pytest.mark.parametrize(
    "dimension",
    (
        Dimension(),
        Dimension(mass=1),
        Dimension(length=1),
        Dimension(time=1),
        Dimension(mass=1, length=1, time=-2),
        Dimension(mass=1, length=2, time=-3),
        Dimension(time=1, current=1),
        Dimension(mass=1, length=2, time=-2, current=-2),
        Dimension(length=-2, luminosity=1),
        Dimension(length=-3, amount=1),
        Dimension(time=-1, currency=1),
    ),
)
def test_dimension_caching(dimension: Dimension) -> None:
    reinstantiation = Dimension(
        mass=dimension.mass,
        length=dimension.length,
        time=dimension.time,
        temperature=dimension.temperature,
        current=dimension.current,
        amount=dimension.amount,
        luminosity=dimension.luminosity,
        currency=dimension.currency,
    )
    assert reinstantiation is dimension


def test_dimension_arithmetic() -> None:
    length = Dimension(length=1)
    time = Dimension(time=1)
    speed = length / time
    assert speed.length == 1
    assert speed.time == -1


@pytest.mark.parametrize(
    ("dimension", "expected"),
    (
        (Dimension(), True),
        (Dimension(length=1, time=-1), False),
        (Dimension(mass=1, length=1, time=-2), False),
        (Dimension(mass=1, length=2, time=-3), False),
        (Dimension(time=1, current=1), False),
        (Dimension(mass=1, length=2, time=-2, current=-2), False),
        (Dimension(length=-2, luminosity=1), False),
        (Dimension(length=-3, amount=1), False),
        (Dimension(time=-1, currency=1), False),
    ),
)
def test_is_dimensionless(dimension: Dimension, expected: bool) -> None:
    assert dimension.is_dimensionless is expected

    assert (dimension / dimension) is Dimension()
    assert (dimension / dimension).is_dimensionless is True


@pytest.mark.parametrize(
    ("dimension", "expected"),
    (
        (Dimension(), False),
        (Dimension(mass=1), True),
        (Dimension(length=1), True),
        (Dimension(time=1), True),
        (Dimension(current=1), True),
        (Dimension(amount=1), True),
        (Dimension(luminosity=1), True),
        (Dimension(currency=1), True),
        (Dimension(length=1, time=-1), False),
        (Dimension(mass=1, length=1, time=-2), False),
        (Dimension(mass=1, length=2, time=-3), False),
        (Dimension(time=1, current=1), False),
        (Dimension(mass=1, length=2, time=-2, current=-2), False),
        (Dimension(length=-2, luminosity=1), False),
        (Dimension(length=-3, amount=1), False),
        (Dimension(time=-1, currency=1), False),
    ),
)
def test_is_primary(dimension: Dimension, expected: bool) -> None:
    assert dimension.is_primary is expected


@pytest.mark.parametrize(
    ("dimension", "expected"),
    (
        (Dimension(), False),
        (Dimension(mass=1), False),
        (Dimension(length=1), False),
        (Dimension(time=1), False),
        (Dimension(current=1), False),
        (Dimension(amount=1), False),
        (Dimension(luminosity=1), False),
        (Dimension(currency=1), False),
        (Dimension(length=1, time=-1), True),
        (Dimension(mass=1, length=1, time=-2), True),
        (Dimension(mass=1, length=2, time=-3), True),
        (Dimension(time=1, current=1), True),
        (Dimension(mass=1, length=2, time=-2, current=-2), True),
        (Dimension(length=-2, luminosity=1), True),
        (Dimension(length=-3, amount=1), True),
        (Dimension(time=-1, currency=1), True),
    ),
)
def test_is_derived(dimension: Dimension, expected: bool) -> None:
    assert dimension.is_derived is expected


@pytest.mark.parametrize(
    ("dimension", "expected"),
    (
        (Dimension(), "Dimension()"),
        (Dimension(length=1, time=-1), "Dimension(length=1, time=-1)"),
        (Dimension(mass=1, length=1, time=-2), "Dimension(mass=1, length=1, time=-2)"),
        (Dimension(mass=1, length=2, time=-3), "Dimension(mass=1, length=2, time=-3)"),
        (Dimension(time=1, current=1), "Dimension(time=1, current=1)"),
        (
            Dimension(mass=1, length=2, time=-2, current=-2),
            "Dimension(mass=1, length=2, time=-2, current=-2)",
        ),
        (Dimension(length=-2, luminosity=1), "Dimension(length=-2, luminosity=1)"),
        (Dimension(length=-3, amount=1), "Dimension(length=-3, amount=1)"),
        (Dimension(time=-1, currency=1), "Dimension(time=-1, currency=1)"),
    ),
)
def test_dimension_repr(dimension: Dimension, expected: str) -> None:
    assert repr(dimension) == expected


@pytest.mark.parametrize(
    ("dimension", "expected"),
    (
        (Dimension(), "dimensionless ([Dimensionless])"),
        (Dimension(length=1, time=-1), "[Length] [Time]^(-1)"),
        (Dimension(mass=1, length=1, time=-2), "force ([Mass] [Length] [Time]^(-2))"),
        (Dimension(mass=1, length=2, time=-3), "power ([Mass] [Length]^2 [Time]^(-3))"),
        (Dimension(time=1, current=1), "charge ([Time] [Current])"),
        (
            Dimension(mass=1, length=2, time=-2, current=-2),
            "[Mass] [Length]^2 [Time]^(-2) [Current]^(-2)",
        ),
        (Dimension(length=-2, luminosity=1), "[Length]^(-2) [Luminosity]"),
        (Dimension(length=-3, amount=1), "[Length]^(-3) [Amount]"),
        (Dimension(time=-1, currency=1), "[Time]^(-1) [Currency]"),
    ),
)
def test_dimension_str(dimension: Dimension, expected: str) -> None:
    assert str(dimension) == expected


co2 = Metadata("co2")
fuel = Metadata("fuel")
biogenic = Metadata("biogenic")


def test_a_tagged_dimension_is_cached() -> None:
    assert MASS * co2 is MASS * co2
    assert (MASS * co2) / (MASS * fuel) is (MASS * co2) / (MASS * fuel)


@pytest.mark.parametrize(
    "other",
    (MASS, MASS * fuel, AMOUNT * co2, (MASS * co2) * biogenic, MASS * co2 / TIME, MASS / TIME * co2),
)
def test_a_tagged_dimension_is_only_itself(other: Dimension) -> None:
    assert MASS * co2 is not other


def test_a_tag_wraps_the_dimension_it_multiplies() -> None:
    assert MASS / TIME * co2 is not MASS * co2 / TIME


def test_tagged_dimension_arithmetic() -> None:
    assert (MASS * co2) / (MASS * fuel) * (MASS * fuel) is MASS * co2
    assert (MASS * co2) / (MASS * co2) is DIMENSIONLESS
    assert (MASS * co2) ** 2 / (MASS * co2) is MASS * co2
    assert (MASS * co2) ** 0 is DIMENSIONLESS
    assert (MASS * co2) * TIME is TIME * (MASS * co2)


def test_a_tagged_dimension_is_not_dimensionless() -> None:
    assert (DIMENSIONLESS * co2).is_dimensionless is False


def test_a_tag_does_not_multiply_a_dimension_from_the_left() -> None:
    with pytest.raises(TypeError):
        _ = co2 * MASS  # pyrefly: ignore[unsupported-operation]  # ty: ignore[unsupported-operator]


@pytest.mark.parametrize(
    ("dimension", "expected_str", "expected_repr"),
    (
        (MASS * co2, "[Mass]|co2", "Dimension(tagged={(Dimension(mass=1), Metadata(value='co2')): 1})"),
        (DIMENSIONLESS * co2, "[Dimensionless]|co2", "Dimension(tagged={(Dimension(), Metadata(value='co2')): 1})"),
        (
            MASS * co2 / TIME,
            "[Time]^(-1) [Mass]|co2",
            "Dimension(time=-1, tagged={(Dimension(mass=1), Metadata(value='co2')): 1})",
        ),
        (
            MASS / TIME * co2,
            "([Mass] [Time]^(-1))|co2",
            "Dimension(tagged={(Dimension(mass=1, time=-1), Metadata(value='co2')): 1})",
        ),
    ),
)
def test_tagged_dimension_representation(dimension: Dimension, expected_str: str, expected_repr: str) -> None:
    assert str(dimension) == expected_str
    assert repr(dimension) == expected_repr


def test_tagged_dimension_str_orders_its_tagged_factors() -> None:
    assert str((MASS * fuel) ** -1 * (MASS * co2)) == "[Mass]|co2 [Mass]|fuel^(-1)"
    assert str((MASS * co2) * biogenic) == "[Mass]|co2|biogenic"

import pytest

from bilt.units import Dimension


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

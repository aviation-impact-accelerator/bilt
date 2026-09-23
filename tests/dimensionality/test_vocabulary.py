"""Tests for the public dimensionality vocabulary."""

import pytest

from bilt.units import Dimension, Unit, dimensions, units
from bilt.units.dimensions import (
    AMOUNT,
    AREA,
    CHARGE,
    CONDUCTANCE,
    CURRENCY,
    CURRENT,
    DIMENSIONLESS,
    DYNAMIC_VISCOSITY,
    ENERGY,
    FORCE,
    LENGTH,
    LUMINOSITY,
    MASS,
    POWER,
    PRESSURE,
    RESISTANCE,
    TEMPERATURE,
    TIME,
    VOLTAGE,
    VOLUME,
    describe,
)

VOCABULARY = (
    (MASS, "mass", units.kilogram),
    (LENGTH, "length", units.meter),
    (TIME, "time", units.second),
    (TEMPERATURE, "temperature", units.kelvin),
    (CURRENT, "current", units.ampere),
    (AMOUNT, "amount", units.mole),
    (LUMINOSITY, "luminosity", units.candela),
    (CURRENCY, "currency", units.USD[2025]),
    (DIMENSIONLESS, "dimensionless", units.radian),
    (AREA, "area", units.hectare),
    (VOLUME, "volume", units.liter),
    (FORCE, "force", units.newton),
    (ENERGY, "energy", units.joule),
    (POWER, "power", units.watt),
    (PRESSURE, "pressure", units.pascal),
    (DYNAMIC_VISCOSITY, "dynamic viscosity", units.poise),
    (CHARGE, "charge", units.coulomb),
    (VOLTAGE, "voltage", units.volt),
    (RESISTANCE, "resistance", units.ohm),
    (CONDUCTANCE, "conductance", units.siemen),
)
IDS = [name for _, name, _ in VOCABULARY]


def _constant_name(name: str) -> str:
    return name.upper().replace(" ", "_")


def test_public_surface_is_exactly_the_vocabulary() -> None:
    constants = {_constant_name(name) for name in IDS}
    assert set(dimensions.__all__) == constants | {"describe"}


@pytest.mark.parametrize(("dimension", "name", "unit"), VOCABULARY, ids=IDS)
def test_named_dimensionality_is_the_dimensionality_of_its_unit(
    dimension: Dimension,
    name: str,
    unit: Unit,
) -> None:
    assert unit.dimension is dimension, f"{name} does not match {unit}"


def test_every_named_unit_has_a_named_dimensionality() -> None:
    for name in units.__all__:
        unit = getattr(units, name)
        if not isinstance(unit, Unit):
            continue
        rendered = describe(unit.dimension)
        assert rendered != str(unit.dimension), f"{name} carries a dimensionality the vocabulary does not name"


@pytest.mark.parametrize(("dimension", "name", "_unit"), VOCABULARY, ids=IDS)
def test_describe_renders_name_and_exponent_form(dimension: Dimension, name: str, _unit: Unit) -> None:
    assert describe(dimension) == f"{name} ({dimension})"


def test_describe_unnamed_dimensionality_is_exponent_form_alone() -> None:
    velocity = LENGTH / TIME
    assert describe(velocity) == str(velocity)

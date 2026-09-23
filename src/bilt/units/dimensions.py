"""Public dimensionality constants for quantity annotations."""

__all__ = [
    "AMOUNT",
    "AREA",
    "CHARGE",
    "CONDUCTANCE",
    "CURRENCY",
    "CURRENT",
    "DIMENSIONLESS",
    "DYNAMIC_VISCOSITY",
    "ENERGY",
    "FORCE",
    "LENGTH",
    "LUMINOSITY",
    "MASS",
    "POWER",
    "PRESSURE",
    "RESISTANCE",
    "TEMPERATURE",
    "TIME",
    "VOLTAGE",
    "VOLUME",
    "describe",
]

from ._dimensions import AMOUNT, CURRENCY, CURRENT, LENGTH, LUMINOSITY, MASS, TEMPERATURE, TIME, Dimension

DIMENSIONLESS = Dimension()
AREA = LENGTH**2
VOLUME = LENGTH**3
FORCE = MASS * LENGTH / TIME**2
ENERGY = MASS * LENGTH**2 / TIME**2
POWER = MASS * LENGTH**2 / TIME**3
PRESSURE = MASS / (LENGTH * TIME**2)
DYNAMIC_VISCOSITY = MASS / (LENGTH * TIME)
CHARGE = TIME * CURRENT
VOLTAGE = MASS * LENGTH**2 / (TIME**3 * CURRENT)
RESISTANCE = MASS * LENGTH**2 / (TIME**3 * CURRENT**2)
CONDUCTANCE = TIME**3 * CURRENT**2 / (MASS * LENGTH**2)


_DIMENSION_NAMES: dict[Dimension, str] = {
    MASS: "mass",
    LENGTH: "length",
    TIME: "time",
    TEMPERATURE: "temperature",
    CURRENT: "current",
    AMOUNT: "amount",
    LUMINOSITY: "luminosity",
    CURRENCY: "currency",
    DIMENSIONLESS: "dimensionless",
    AREA: "area",
    VOLUME: "volume",
    FORCE: "force",
    ENERGY: "energy",
    POWER: "power",
    PRESSURE: "pressure",
    DYNAMIC_VISCOSITY: "dynamic viscosity",
    CHARGE: "charge",
    VOLTAGE: "voltage",
    RESISTANCE: "resistance",
    CONDUCTANCE: "conductance",
}


def describe(dimension: Dimension) -> str:
    """Render a dimensionality as its name followed by its exponent form.

    Args:
        dimension: The dimensionality to render.

    Returns:
        The name and the exponent form together, as in
        ``"energy ([Mass] [Length]^2 [Time]^(-2))"``. A dimensionality this
        module does not name renders as its exponent form alone.
    """
    name = _DIMENSION_NAMES.get(dimension)
    if name is None:
        return str(dimension)
    return f"{name} ({dimension})"

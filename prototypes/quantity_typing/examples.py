"""Executable models: no casts, Any annotations or checker ignores."""

from typing import Annotated, assert_type
from bilt_typing import (
    Implementation,
    Interface,
    Tensor,
    Float,
    Requires,
    dim,
    equation,
    equal,
    field,
    pointwise,
    system,
    tensor,
    Quantity,
    transform,
)
from bilt_typing.quantities import Mass, Length, Acceleration, Force, AbsoluteTemperature, MassPerLength
from bilt_typing import dimension_types as dt
from bilt_typing.units import (
    kilogram,
    gram,
    meter,
    meter_per_second_squared,
    absolute_celsius,
    absolute_fahrenheit,
    kilogram_per_meter,
)

AIRCRAFT = dim("aircraft")
YEAR = dim("year")
type AircraftMass = Annotated[Mass[Tensor[float]], Requires(AIRCRAFT)]


@equation
def doubling(m: Mass[float], doubled: Mass[float]) -> bool:
    return doubled == 2.0 * m


@system
class Scale(metaclass=Interface):
    m: Mass[Tensor[float]] = field()
    doubled: Mass[Tensor[float]] = field()


@system
class ScalarScale(metaclass=Interface):
    m: Mass[float] = field()
    doubled: Mass[float] = field()


@equation
def same_temperature(same: AbsoluteTemperature[float], t: AbsoluteTemperature[float]) -> bool:
    return same == t


@system
class AbsoluteThermometer(metaclass=Interface):
    t: AbsoluteTemperature[float] = field()
    same: AbsoluteTemperature[float] = field()


@equation
def newtons_law(mass: Mass[float], acceleration: Acceleration[float], force: Force[float]) -> bool:
    return equal(force, mass * acceleration)


@system
class Fleet(metaclass=Interface):
    mass: Mass[Tensor[float]] = field()
    acceleration: Acceleration[float] = field()
    force: Force[Tensor[float]] = field()


@equation
def total_mass(masses: AircraftMass, total: Mass[Float]) -> bool:
    return equal(total, masses.sum(AIRCRAFT))


@system
class FleetTotal(metaclass=Interface):
    masses: AircraftMass = field()
    total: Mass[Float] = field()


@pointwise
def doubled_pointwise(m: Mass[float]) -> Mass[float]:
    return 2.0 * m


@equation
def cable_mass(density: MassPerLength[float], length: Length[float], mass: Mass[float]) -> bool:
    return equal(mass, density * length)


@system
class Cable(metaclass=Interface):
    density: MassPerLength[float] = field()
    length: Length[float] = field()
    mass: Mass[float] = field()


def _doubled(m: Mass[Tensor[float]]) -> Mass[Tensor[float]]:
    with Implementation(Scale, [doubling]):
        return Scale(m=m).doubled


def run() -> None:
    scalar = 3.0 * gram
    assert_type(scalar, Quantity[dt.MASS, float])
    assert_type(doubled_pointwise(scalar), Quantity[dt.MASS, float])
    masses = tensor([1.0, 2.0, 3.0], over=AIRCRAFT, unit=kilogram)
    assert_type(masses, Quantity[dt.MASS, Tensor[float]])
    assert_type(_doubled(masses), Quantity[dt.MASS, Tensor[float]])
    print("Tensor scale:", _doubled(masses))
    with Implementation(AbsoluteThermometer, [same_temperature]):
        print("Temperature:", AbsoluteThermometer(t=32.0 * absolute_fahrenheit).same.convert_to(absolute_celsius))
    with Implementation(Fleet, [newtons_law]):
        result = Fleet(mass=masses, acceleration=2.0 * meter_per_second_squared).force
        assert_type(result, Quantity[dt.FORCE, Tensor[float]])
        print("Mixed scalar/tensor force:", result)
    with Implementation(FleetTotal, [total_mass]):
        result_total = FleetTotal(masses=masses).total
        assert_type(result_total, Quantity[dt.MASS, Float])
        print("Reduction:", result_total)
    with Implementation(Cable, [cable_mass]):
        print("Custom dimension:", Cable(density=2.0 * kilogram_per_meter, length=3.0 * meter).mass)


if __name__ == "__main__":
    run()


@transform
def total(masses: AircraftMass) -> Mass[Float]:
    return masses.sum(AIRCRAFT)

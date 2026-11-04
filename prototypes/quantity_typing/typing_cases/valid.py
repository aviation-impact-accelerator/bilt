"""Check inference for arithmetic, nominal identity, lifting and custom dimensions."""

from typing import assert_type
from bilt_typing import Quantity, Tensor, Float, tensor, equal
from bilt_typing import dimension_types as dt
from bilt_typing.units import kilogram, gram, meter, meter_per_second_squared, kilogram_per_meter
from examples import AIRCRAFT, doubled_pointwise

mass = 2.0 * kilogram
length = 3.0 * meter
acceleration = 4.0 * meter_per_second_squared
masses = tensor([1.0, 2.0], over=AIRCRAFT, unit=kilogram)
accelerations = tensor([3.0, 4.0], over=AIRCRAFT, unit=meter_per_second_squared)
assert_type(mass * acceleration, Quantity[dt.FORCE, float])
assert_type(masses * acceleration, Quantity[dt.FORCE, Tensor[float]])
assert_type(mass * accelerations, Quantity[dt.FORCE, Tensor[float]])
assert_type(masses * accelerations, Quantity[dt.FORCE, Tensor[float]])
assert_type(length * length, Quantity[dt.AREA, float])
assert_type(mass / length, Quantity[dt.MASS_PER_LENGTH, float])
assert_type((2.0 * kilogram_per_meter) * length, Quantity[dt.MASS, float])
assert_type(mass.convert_to(gram), Quantity[dt.MASS, float])
assert_type(masses.convert_to(gram), Quantity[dt.MASS, Tensor[float]])
assert_type(mass + masses, Quantity[dt.MASS, Tensor[float]])
assert_type(masses.sum(AIRCRAFT), Quantity[dt.MASS, Float])
assert_type(doubled_pointwise(masses), Quantity[dt.MASS, Tensor[float]])
assert_type(equal(mass, masses), bool)

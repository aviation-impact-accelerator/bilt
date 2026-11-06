"""Each marked line must be rejected by all three checkers; no suppressions."""

from examples import Scale, ScalarScale, doubling, AIRCRAFT
from bilt_typing import equal, tensor
from bilt_typing.quantities import Mass
from bilt_typing.units import kilogram, meter

scalar = 2.0 * kilogram
length = 3.0 * meter
vector = tensor([1.0, 2.0], over=AIRCRAFT, unit=kilogram)
Scale(m=scalar)  # error
ScalarScale(m=vector)  # error
ScalarScale(m=length)  # error
wrong: Mass[float] = vector  # error
equal(scalar, length)  # error
scalar.convert_to(meter)  # error
doubling(m=length, doubled=scalar)  # error
scalar.sum(AIRCRAFT)  # error

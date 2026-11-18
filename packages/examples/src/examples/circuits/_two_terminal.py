"""Equations shared by every two-terminal component with `positive` and `negative` pins.

Sign convention (passive): `current` enters at the `positive` pin, flows through the
component and leaves at the `negative` pin. `voltage` is the potential at `positive`
minus the potential at `negative`.

Equations bind by port name, so any component whose interface uses these port names
can include these equations in its implementation.
"""

from typing import Annotated

from bilt import Quantity, equation
from bilt.units.dimensions import CURRENT, VOLTAGE
from examples.circuits._pin import Pin


@equation
def potential_difference(positive: Pin, negative: Pin, voltage: Annotated[Quantity, VOLTAGE]) -> bool:
    """Voltage across the component."""
    return voltage == positive.potential - negative.potential


@equation
def current_conservation(positive: Pin, negative: Pin) -> bool:
    """Current entering at the positive pin leaves at the negative pin."""
    return positive.current == -negative.current


@equation
def current_through(positive: Pin, current: Annotated[Quantity, CURRENT]) -> bool:
    """Current through the component is the current entering at the positive pin."""
    return current == positive.current

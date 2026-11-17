"""Ideal voltage source."""

from typing import Annotated

from bilt import Implementation, Interface, Quantity, QuantityTensor, field, system
from bilt.units.dimensions import CURRENT, VOLTAGE
from examples.circuits._two_terminal import potential_difference


@system
class VoltageSource(metaclass=Interface):
    """Ideal source: `voltage` is the EMF, positive terminal at higher potential."""

    positive_potential: Annotated[Quantity | QuantityTensor, VOLTAGE] = field()
    negative_potential: Annotated[Quantity | QuantityTensor, VOLTAGE] = field()
    voltage: Annotated[Quantity | QuantityTensor, VOLTAGE] = field()
    current: Annotated[Quantity | QuantityTensor, CURRENT] = field()


voltage_source = Implementation(VoltageSource, [potential_difference])

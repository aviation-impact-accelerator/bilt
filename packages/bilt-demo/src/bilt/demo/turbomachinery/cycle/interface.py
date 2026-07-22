import typing

import bilt.core
from bilt.demo.turbomachinery.fluid.interface import FluidProperties
from bilt.units import DIMENSIONLESS, fraction, joule, kilogram, second


@bilt.core.system
class Cycle(metaclass=bilt.core.Interface):
    fluid: FluidProperties
    air_mass_flow_rate: typing.Annotated[float, kilogram / second]
    net_specific_work: typing.Annotated[float, joule / kilogram]
    thermal_efficiency: typing.Annotated[float, fraction]
    pressure_ratio: typing.Annotated[float, DIMENSIONLESS]

import typing

import bilt
from bilt.demo.fluid.interface import FluidProperties


@bilt.core.system
class Cycle:
    fluid: FluidProperties
    air_mass_flow_rate: typing.Annotated[float, bilt.unit.kilogram / bilt.unit.second]
    net_specific_work: typing.Annotated[float, bilt.unit.joule / bilt.unit.kilogram]
    thermal_efficiency: typing.Annotated[float, bilt.unit.fraction]
    pressure_ratio: typing.Annotated[float, bilt.unit.DIMENSIONLESS]

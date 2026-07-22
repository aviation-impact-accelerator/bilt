import typing

import bilt.core
from bilt.demo.turbomachinery.fluid.interface import FluidProperties, FuelFlow, ThermodynamicState
from bilt.units import fraction, joule, kilogram, second


class WorkProducing(typing.Protocol):
    specific_work: typing.Annotated[float, joule / kilogram]


@bilt.core.system
class Compressor(metaclass=bilt.core.Interface):
    fluid: FluidProperties
    inlet: ThermodynamicState
    outlet: ThermodynamicState
    isentropic_efficiency: typing.Annotated[float, fraction]
    specific_work: typing.Annotated[float, joule / kilogram]


@bilt.core.system
class Combustor(metaclass=bilt.core.Interface):
    fluid: FluidProperties
    inlet: ThermodynamicState
    outlet: ThermodynamicState
    fuel_flow: FuelFlow
    air_mass_flow_rate: typing.Annotated[float, kilogram / second]
    pressure_recovery_factor: typing.Annotated[float, fraction]
    heat_addition: typing.Annotated[float, joule / second]


@bilt.core.system
class Turbine(metaclass=bilt.core.Interface):
    fluid: FluidProperties
    inlet: ThermodynamicState
    outlet: ThermodynamicState
    isentropic_efficiency: typing.Annotated[float, fraction]
    specific_work: typing.Annotated[float, joule / kilogram]

import typing

import bilt

from examples.interface.fluid import FluidProperties, FuelFlow, ThermodynamicState


class WorkProducing(typing.Protocol):
    specific_work: typing.Annotated[float, bilt.unit.joule / bilt.unit.kilogram]


@bilt.core.system
class Compressor:
    fluid: FluidProperties
    inlet: ThermodynamicState
    outlet: ThermodynamicState
    isentropic_efficiency: typing.Annotated[float, bilt.unit.fraction]
    specific_work: typing.Annotated[float, bilt.unit.joule / bilt.unit.kilogram]


@bilt.core.system
class Combustor:
    fluid: FluidProperties
    inlet: ThermodynamicState
    outlet: ThermodynamicState
    fuel_flow: FuelFlow
    air_mass_flow_rate: typing.Annotated[float, bilt.unit.kilogram / bilt.unit.second]
    pressure_recovery_factor: typing.Annotated[float, bilt.unit.fraction]
    heat_addition: typing.Annotated[float, bilt.unit.joule / bilt.unit.second]


@bilt.core.system
class Turbine:
    fluid: FluidProperties
    inlet: ThermodynamicState
    outlet: ThermodynamicState
    isentropic_efficiency: typing.Annotated[float, bilt.unit.fraction]
    specific_work: typing.Annotated[float, bilt.unit.joule / bilt.unit.kilogram]

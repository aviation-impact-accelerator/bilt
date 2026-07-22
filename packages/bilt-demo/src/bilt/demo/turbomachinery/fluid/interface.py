import typing

import bilt.core
from bilt.units import DIMENSIONLESS, fraction, joule, kelvin, kilogram, meter, pascal, second


@bilt.core.system
class FluidProperties(metaclass=bilt.core.Interface):
    specific_heat_capacity: typing.Annotated[float, joule / (kilogram * kelvin)]
    heat_capacity_ratio: typing.Annotated[float, DIMENSIONLESS]
    specific_gas_constant: typing.Annotated[float, joule / (kilogram * kelvin)]


@bilt.core.system
class ThermodynamicState(metaclass=bilt.core.Interface):
    fluid: FluidProperties
    temperature: typing.Annotated[float, kelvin]
    pressure: typing.Annotated[float, pascal]
    specific_enthalpy: typing.Annotated[float, joule / kilogram]
    density: typing.Annotated[float, kilogram / meter**3]


@bilt.core.system
class FuelProperties(metaclass=bilt.core.Interface):
    lower_heating_value: typing.Annotated[float, joule / kilogram]
    combustion_efficiency: typing.Annotated[float, fraction]


@bilt.core.system
class FuelFlow(metaclass=bilt.core.Interface):
    fuel: FuelProperties
    mass_flow_rate: typing.Annotated[float, kilogram / second]
    heat_release: typing.Annotated[float, joule / second]

import typing

import bilt


@bilt.core.system
class FluidProperties:
    specific_heat_capacity: typing.Annotated[float, bilt.unit.joule / (bilt.unit.kilogram * bilt.unit.kelvin)]
    heat_capacity_ratio: typing.Annotated[float, bilt.unit.DIMENSIONLESS]
    specific_gas_constant: typing.Annotated[float, bilt.unit.joule / (bilt.unit.kilogram * bilt.unit.kelvin)]


@bilt.core.system
class ThermodynamicState:
    fluid: FluidProperties
    temperature: typing.Annotated[float, bilt.unit.kelvin]
    pressure: typing.Annotated[float, bilt.unit.pascal]
    specific_enthalpy: typing.Annotated[float, bilt.unit.joule / bilt.unit.kilogram]
    density: typing.Annotated[float, bilt.unit.kilogram / bilt.unit.metre**3]


@bilt.core.system
class FuelProperties:
    lower_heating_value: typing.Annotated[float, bilt.unit.joule / bilt.unit.kilogram]
    combustion_efficiency: typing.Annotated[float, bilt.unit.fraction]


@bilt.core.system
class FuelFlow:
    fuel: FuelProperties
    mass_flow_rate: typing.Annotated[float, bilt.unit.kilogram / bilt.unit.second]
    heat_release: typing.Annotated[float, bilt.unit.joule / bilt.unit.second]

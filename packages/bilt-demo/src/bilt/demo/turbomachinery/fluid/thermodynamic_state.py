import typing

import bilt.core
from bilt.demo.turbomachinery.fluid.interface import FluidProperties, ThermodynamicState
from bilt.units import joule, kelvin, kilogram, meter, pascal


@bilt.core.equation
def specific_enthalpy_relation(
    fluid: FluidProperties,
    temperature: typing.Annotated[float, kelvin],
    specific_enthalpy: typing.Annotated[float, joule / kilogram],
) -> bool:
    return specific_enthalpy == fluid.specific_heat_capacity * temperature


@bilt.core.equation
def density_relation(
    fluid: FluidProperties,
    temperature: typing.Annotated[float, kelvin],
    pressure: typing.Annotated[float, pascal],
    density: typing.Annotated[float, kilogram / meter**3],
) -> bool:
    return density == pressure / (fluid.specific_gas_constant * temperature)


ideal_gas_state = bilt.core.Implementation(
    ThermodynamicState,
    {specific_enthalpy_relation, density_relation},
)

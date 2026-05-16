import typing

import bilt

from examples.interface.fluid import FluidProperties, ThermodynamicState


@bilt.core.equation
def specific_enthalpy_relation(
    fluid: FluidProperties,
    temperature: typing.Annotated[float, bilt.unit.kelvin],
    specific_enthalpy: typing.Annotated[float, bilt.unit.joule / bilt.unit.kilogram],
) -> bilt.core.equation:
    return specific_enthalpy == fluid.specific_heat_capacity * temperature


@bilt.core.equation
def density_relation(
    fluid: FluidProperties,
    temperature: typing.Annotated[float, bilt.unit.kelvin],
    pressure: typing.Annotated[float, bilt.unit.pascal],
    density: typing.Annotated[float, bilt.unit.kilogram / bilt.unit.metre**3],
) -> bilt.core.equation:
    return density == pressure / (fluid.specific_gas_constant * temperature)


ideal_gas_state = bilt.Implementation(
    ThermodynamicState,
    {specific_enthalpy_relation, density_relation},
)

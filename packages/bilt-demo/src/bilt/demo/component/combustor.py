import typing

import bilt
from bilt.demo.component.interface import Combustor
from bilt.demo.fluid.interface import FluidProperties, FuelFlow, ThermodynamicState


@bilt.core.equation
def heat_addition_relation(
    fuel_flow: FuelFlow,
    heat_addition: typing.Annotated[float, bilt.unit.joule / bilt.unit.second],
) -> bilt.core.equation:
    return heat_addition == fuel_flow.heat_release


@bilt.core.equation
def outlet_pressure_relation(
    inlet: ThermodynamicState,
    pressure_recovery_factor: typing.Annotated[float, bilt.unit.fraction],
    outlet: ThermodynamicState,
) -> bilt.core.equation:
    return outlet.pressure == pressure_recovery_factor * inlet.pressure


@bilt.core.equation
def outlet_temperature_relation(
    inlet: ThermodynamicState,
    fluid: FluidProperties,
    heat_addition: typing.Annotated[float, bilt.unit.joule / bilt.unit.second],
    air_mass_flow_rate: typing.Annotated[float, bilt.unit.kilogram / bilt.unit.second],
    outlet: ThermodynamicState,
) -> bilt.core.equation:
    return outlet.temperature == inlet.temperature + heat_addition / (air_mass_flow_rate * fluid.specific_heat_capacity)


@bilt.core.equation
def inlet_fluid_equals_combustor_fluid(
    fluid: FluidProperties,
    inlet: ThermodynamicState,
) -> bilt.core.equation:
    return fluid == inlet.fluid


@bilt.core.equation
def outlet_fluid_equals_combustor_fluid(
    fluid: FluidProperties,
    outlet: ThermodynamicState,
) -> bilt.core.equation:
    return fluid == outlet.fluid


simple_combustor = bilt.Implementation(
    Combustor,
    {
        heat_addition_relation,
        outlet_pressure_relation,
        outlet_temperature_relation,
        inlet_fluid_equals_combustor_fluid,
        outlet_fluid_equals_combustor_fluid,
    },
)

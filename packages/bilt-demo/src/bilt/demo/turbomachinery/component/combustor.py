import typing

import bilt.core
from bilt.demo.turbomachinery.component.interface import Combustor
from bilt.demo.turbomachinery.fluid.interface import FluidProperties, FuelFlow, ThermodynamicState
from bilt.units import fraction, joule, kilogram, second


@bilt.core.equation
def heat_addition_relation(
    fuel_flow: FuelFlow,
    heat_addition: typing.Annotated[float, joule / second],
) -> bool:
    return heat_addition == fuel_flow.heat_release


@bilt.core.equation
def outlet_pressure_relation(
    inlet: ThermodynamicState,
    pressure_recovery_factor: typing.Annotated[float, fraction],
    outlet: ThermodynamicState,
) -> bool:
    return outlet.pressure == pressure_recovery_factor * inlet.pressure


@bilt.core.equation
def outlet_temperature_relation(
    inlet: ThermodynamicState,
    fluid: FluidProperties,
    heat_addition: typing.Annotated[float, joule / second],
    air_mass_flow_rate: typing.Annotated[float, kilogram / second],
    outlet: ThermodynamicState,
) -> bool:
    return outlet.temperature == inlet.temperature + heat_addition / (air_mass_flow_rate * fluid.specific_heat_capacity)


@bilt.core.equation
def inlet_fluid_equals_combustor_fluid(
    fluid: FluidProperties,
    inlet: ThermodynamicState,
) -> bool:
    return fluid == inlet.fluid


@bilt.core.equation
def outlet_fluid_equals_combustor_fluid(
    fluid: FluidProperties,
    outlet: ThermodynamicState,
) -> bool:
    return fluid == outlet.fluid


simple_combustor = bilt.core.Implementation(
    Combustor,
    {
        heat_addition_relation,
        outlet_pressure_relation,
        outlet_temperature_relation,
        inlet_fluid_equals_combustor_fluid,
        outlet_fluid_equals_combustor_fluid,
    },
)

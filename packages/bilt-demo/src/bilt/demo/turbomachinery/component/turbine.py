import typing

import bilt.core
from bilt.demo.turbomachinery.component.interface import Turbine
from bilt.demo.turbomachinery.fluid.interface import FluidProperties, ThermodynamicState
from bilt.units import DIMENSIONLESS, fraction, joule, kilogram


@bilt.core.equation
def isentropic_relation(
    inlet: ThermodynamicState,
    outlet: ThermodynamicState,
    heat_capacity_ratio: typing.Annotated[float, DIMENSIONLESS],
    isentropic_efficiency: typing.Annotated[float, fraction],
) -> bool:
    return (
        outlet.temperature / inlet.temperature
        == (outlet.pressure / inlet.pressure) ** ((heat_capacity_ratio - 1) / heat_capacity_ratio)
        * isentropic_efficiency
    )


@bilt.core.equation
def specific_work_relation(
    inlet: ThermodynamicState,
    outlet: ThermodynamicState,
    fluid: FluidProperties,
    specific_work: typing.Annotated[float, joule / kilogram],
) -> bool:
    return specific_work == fluid.specific_heat_capacity * (inlet.temperature - outlet.temperature)


@bilt.core.equation
def inlet_fluid_equals_turbine_fluid(
    fluid: FluidProperties,
    inlet: ThermodynamicState,
) -> bool:
    return fluid == inlet.fluid


@bilt.core.equation
def outlet_fluid_equals_turbine_fluid(
    fluid: FluidProperties,
    outlet: ThermodynamicState,
) -> bool:
    return fluid == outlet.fluid


isentropic_turbine = bilt.core.Implementation(
    Turbine,
    {
        isentropic_relation,
        specific_work_relation,
        inlet_fluid_equals_turbine_fluid,
        outlet_fluid_equals_turbine_fluid,
    },
)

import typing

import bilt

from examples.interface.component import Turbine
from examples.interface.fluid import FluidProperties, ThermodynamicState


@bilt.core.equation
def isentropic_relation(
    inlet: ThermodynamicState,
    outlet: ThermodynamicState,
    heat_capacity_ratio: typing.Annotated[float, bilt.unit.DIMENSIONLESS],
    isentropic_efficiency: typing.Annotated[float, bilt.unit.fraction],
) -> bilt.core.equation:
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
    specific_work: typing.Annotated[float, bilt.unit.joule / bilt.unit.kilogram],
) -> bilt.core.equation:
    return specific_work == fluid.specific_heat_capacity * (inlet.temperature - outlet.temperature)


@bilt.core.equation
def inlet_fluid_equals_turbine_fluid(
    fluid: FluidProperties,
    inlet: ThermodynamicState,
) -> bilt.core.equation:
    return fluid == inlet.fluid


@bilt.core.equation
def outlet_fluid_equals_turbine_fluid(
    fluid: FluidProperties,
    outlet: ThermodynamicState,
) -> bilt.core.equation:
    return fluid == outlet.fluid


isentropic_turbine = bilt.Implementation(
    Turbine,
    {
        isentropic_relation,
        specific_work_relation,
        inlet_fluid_equals_turbine_fluid,
        outlet_fluid_equals_turbine_fluid,
    },
)

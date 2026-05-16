import typing

import bilt
from bilt.demo.component.interface import Combustor, Compressor, Turbine
from bilt.demo.cycle.common import net_specific_work, pressure_ratio_relation, thermal_efficiency_relation
from bilt.demo.cycle.interface import Cycle
from bilt.demo.fluid.interface import FluidProperties


@bilt.core.equation
def compressor_outlet_feeds_combustor_inlet(
    compressor: Compressor,
    combustor: Combustor,
) -> bilt.core.equation:
    return compressor.outlet == combustor.inlet


@bilt.core.equation
def combustor_outlet_feeds_turbine_inlet(
    combustor: Combustor,
    turbine: Turbine,
) -> bilt.core.equation:
    return combustor.outlet == turbine.inlet


@bilt.core.equation
def compressor_fluid_equals_cycle_fluid(
    fluid: FluidProperties,
    compressor: Compressor,
) -> bilt.core.equation:
    return fluid == compressor.fluid


@bilt.core.equation
def combustor_fluid_equals_cycle_fluid(
    fluid: FluidProperties,
    combustor: Combustor,
) -> bilt.core.equation:
    return fluid == combustor.fluid


@bilt.core.equation
def turbine_fluid_equals_cycle_fluid(
    fluid: FluidProperties,
    turbine: Turbine,
) -> bilt.core.equation:
    return fluid == turbine.fluid


@bilt.core.equation
def air_mass_flow_rate_feeds_combustor(
    air_mass_flow_rate: typing.Annotated[float, bilt.unit.kilogram / bilt.unit.second],
    combustor: Combustor,
) -> bilt.core.equation:
    return air_mass_flow_rate == combustor.air_mass_flow_rate


simple_brayton_cycle = bilt.Implementation(
    Cycle,
    {
        net_specific_work,
        thermal_efficiency_relation,
        pressure_ratio_relation,
        compressor_outlet_feeds_combustor_inlet,
        combustor_outlet_feeds_turbine_inlet,
        compressor_fluid_equals_cycle_fluid,
        combustor_fluid_equals_cycle_fluid,
        turbine_fluid_equals_cycle_fluid,
        air_mass_flow_rate_feeds_combustor,
    },
)

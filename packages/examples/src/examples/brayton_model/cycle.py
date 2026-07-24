from __future__ import annotations

import bilt

from .combustor import simple_combustor
from .compressor import isentropic_compressor
from .interfaces import (
    BraytonCycle,
    Combustor,
    Compressor,
    FluidProperties,
    FuelProperties,
    ThermodynamicState,
    Turbine,
)
from .turbine import isentropic_turbine

# Public cycle boundary -> private component boundary.


@bilt.core.transform
def specific_heat_capacity(
    fluid: FluidProperties,
) -> float:
    return fluid.specific_heat_capacity


@bilt.core.transform
def heat_capacity_ratio(
    fluid: FluidProperties,
) -> float:
    return fluid.heat_capacity_ratio


@bilt.core.transform
def lower_heating_value(
    fuel: FuelProperties,
) -> float:
    return fuel.lower_heating_value


@bilt.core.transform
def combustion_efficiency(
    fuel: FuelProperties,
) -> float:
    return fuel.combustion_efficiency


@bilt.core.transform
def inlet_temperature(
    inlet: ThermodynamicState,
) -> float:
    return inlet.temperature


@bilt.core.transform
def inlet_pressure(
    inlet: ThermodynamicState,
) -> float:
    return inlet.pressure


@bilt.core.transform
def compressor_outlet_pressure(
    pressure_ratio: float,
    compressor: Compressor,
) -> float:
    return pressure_ratio * compressor.inlet.pressure


@bilt.core.transform
def combustor_inlet_temperature(
    compressor: Compressor,
) -> float:
    return compressor.outlet.temperature


@bilt.core.transform
def combustor_inlet_pressure(
    compressor: Compressor,
) -> float:
    return compressor.outlet.pressure


@bilt.core.transform
def air_mass_flow(
    air_mass_flow_rate: float,
) -> float:
    return air_mass_flow_rate


@bilt.core.transform
def fuel_mass_flow(
    fuel_mass_flow_rate: float,
) -> float:
    return fuel_mass_flow_rate


@bilt.core.transform
def turbine_inlet_temperature(
    combustor: Combustor,
) -> float:
    return combustor.outlet.temperature


@bilt.core.transform
def turbine_inlet_pressure(
    combustor: Combustor,
) -> float:
    return combustor.outlet.pressure


@bilt.core.transform
def turbine_outlet_pressure(
    inlet: ThermodynamicState,
) -> float:
    return inlet.pressure


# Private component results -> public cycle outputs.


@bilt.core.transform
def net_specific_work(
    compressor: Compressor,
    turbine: Turbine,
) -> float:
    return turbine.specific_work - compressor.specific_work


@bilt.core.transform
def shaft_power(
    net_specific_work: float,
    air_mass_flow_rate: float,
) -> float:
    return net_specific_work * air_mass_flow_rate


@bilt.core.transform
def thermal_efficiency(
    shaft_power: float,
    combustor: Combustor,
) -> float:
    return shaft_power / combustor.heat_addition


@bilt.core.transform
def exhaust_temperature(
    turbine: Turbine,
) -> float:
    return turbine.outlet.temperature


@bilt.core.transform
def exhaust_pressure(
    turbine: Turbine,
) -> float:
    return turbine.outlet.pressure


@bilt.core.transform
def waste_heat(
    shaft_power: float,
    combustor: Combustor,
) -> float:
    return combustor.heat_addition - shaft_power


def simple_brayton_cycle(
    *,
    compressor_efficiency: float,
    combustor_pressure_recovery: float,
    turbine_efficiency: float,
) -> bilt.core.Implementation:
    """Compose one private compressor-combustor-turbine topology."""
    compressor = isentropic_compressor(
        efficiency=compressor_efficiency,
    )
    combustor = simple_combustor(
        pressure_recovery_factor=combustor_pressure_recovery,
    )
    turbine = isentropic_turbine(
        efficiency=turbine_efficiency,
    )

    return bilt.core.Implementation(
        BraytonCycle,
        {
            # Cycle fluid -> component fluids.
            "compressor.fluid.specific_heat_capacity": specific_heat_capacity,
            "compressor.fluid.heat_capacity_ratio": heat_capacity_ratio,
            "combustor.fluid.specific_heat_capacity": specific_heat_capacity,
            "combustor.fluid.heat_capacity_ratio": heat_capacity_ratio,
            "turbine.fluid.specific_heat_capacity": specific_heat_capacity,
            "turbine.fluid.heat_capacity_ratio": heat_capacity_ratio,
            # Cycle fuel -> combustor fuel.
            "combustor.fuel.lower_heating_value": lower_heating_value,
            "combustor.fuel.combustion_efficiency": combustion_efficiency,
            # Cycle inlet -> compressor inlet.
            "compressor.inlet.temperature": inlet_temperature,
            "compressor.inlet.pressure": inlet_pressure,
            # Compressor pressure boundary.
            "compressor.outlet.pressure": compressor_outlet_pressure,
            # Compressor -> combustor.
            "combustor.inlet.temperature": combustor_inlet_temperature,
            "combustor.inlet.pressure": combustor_inlet_pressure,
            "combustor.air_mass_flow_rate": air_mass_flow,
            "combustor.fuel_mass_flow_rate": fuel_mass_flow,
            # Combustor -> turbine.
            "turbine.inlet.temperature": turbine_inlet_temperature,
            "turbine.inlet.pressure": turbine_inlet_pressure,
            # Expansion boundary.
            "turbine.outlet.pressure": turbine_outlet_pressure,
            # Public cycle outputs.
            "net_specific_work": net_specific_work,
            "shaft_power": shaft_power,
            "thermal_efficiency": thermal_efficiency,
            "exhaust_temperature": exhaust_temperature,
            "exhaust_pressure": exhaust_pressure,
            "waste_heat": waste_heat,
        },
        internal_systems={
            "compressor": compressor,
            "combustor": combustor,
            "turbine": turbine,
        },
    )

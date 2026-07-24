from __future__ import annotations

import bilt

from .fuel_flow import simple_fuel_flow
from .interfaces import (
    Combustor,
    FuelFlow,
    FuelProperties,
    FluidProperties,
    ThermodynamicState,
)


@bilt.core.transform
def fuel_lower_heating_value(
    fuel: FuelProperties,
) -> float:
    return fuel.lower_heating_value


@bilt.core.transform
def fuel_combustion_efficiency(
    fuel: FuelProperties,
) -> float:
    return fuel.combustion_efficiency


@bilt.core.transform
def fuel_mass_flow(
    fuel_mass_flow_rate: float,
) -> float:
    return fuel_mass_flow_rate


@bilt.core.transform
def heat_addition(
    fuel_flow: FuelFlow,
) -> float:
    return fuel_flow.heat_release


@bilt.core.transform
def outlet_pressure(
    inlet: ThermodynamicState,
    pressure_recovery_factor: float,
) -> float:
    return pressure_recovery_factor * inlet.pressure


@bilt.core.transform
def outlet_temperature(
    fluid: FluidProperties,
    inlet: ThermodynamicState,
    heat_addition: float,
    air_mass_flow_rate: float,
) -> float:
    return (
        inlet.temperature
        + heat_addition
        / (
            air_mass_flow_rate
            * fluid.specific_heat_capacity
        )
    )


def simple_combustor(
    *,
    pressure_recovery_factor: float,
) -> bilt.core.Implementation:
    return bilt.core.Implementation(
        Combustor,
        {
            "fuel_flow.fuel.lower_heating_value":
                fuel_lower_heating_value,
            "fuel_flow.fuel.combustion_efficiency":
                fuel_combustion_efficiency,
            "fuel_flow.mass_flow_rate":
                fuel_mass_flow,
            "heat_addition":
                heat_addition,
            "outlet.pressure":
                outlet_pressure,
            "outlet.temperature":
                outlet_temperature,
        },
        internal_systems={
            "fuel_flow": simple_fuel_flow,
        },
        internal_values={
            "pressure_recovery_factor":
                pressure_recovery_factor,
        },
    )

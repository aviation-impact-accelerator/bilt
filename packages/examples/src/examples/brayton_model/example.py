from __future__ import annotations

from examples.brayton_model.cycle import simple_brayton_cycle
from examples.brayton_model.interfaces import (
    BraytonCycle,
    FluidProperties,
    FuelProperties,
    ThermodynamicState,
)


def example() -> BraytonCycle:
    implementation = simple_brayton_cycle(
        compressor_efficiency=0.85,
        combustor_pressure_recovery=0.95,
        turbine_efficiency=0.90,
    )

    with implementation:
        cycle = BraytonCycle(
            fluid=FluidProperties(
                specific_heat_capacity=1_005.0,
                heat_capacity_ratio=1.4,
            ),
            fuel=FuelProperties(
                lower_heating_value=43_000_000.0,
                combustion_efficiency=0.99,
            ),
            inlet=ThermodynamicState(
                temperature=288.15,
                pressure=101_325.0,
            ),
            air_mass_flow_rate=20.0,
            fuel_mass_flow_rate=0.5,
            pressure_ratio=10.0,
        )

        _ = cycle.net_specific_work
        _ = cycle.shaft_power
        _ = cycle.thermal_efficiency
        _ = cycle.exhaust_temperature
        _ = cycle.exhaust_pressure
        _ = cycle.waste_heat

        return cycle


if __name__ == "__main__":
    print(example())

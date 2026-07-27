from __future__ import annotations

from examples.brayton_model.interfaces import (
    Combustor,
    Compressor,
    FluidProperties,
    FuelProperties,
    ThermodynamicState,
    Turbine,
)
from examples.brayton_model.public_cycle import BraytonAssembly, public_brayton_cycle


def public_example() -> BraytonAssembly:
    implementation = public_brayton_cycle(
        compressor_efficiency=0.85,
        combustor_pressure_recovery=0.95,
        turbine_efficiency=0.90,
    )

    with implementation:
        cycle = BraytonAssembly(
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

            # The component instances are now part of the public model.
            # Their unresolved fields are supplied by the parent topology or
            # by their registered child implementations.
            compressor=Compressor(
                fluid=FluidProperties(),
                inlet=ThermodynamicState(),
                outlet=ThermodynamicState(),
            ),
            combustor=Combustor(
                fluid=FluidProperties(),
                fuel=FuelProperties(),
                inlet=ThermodynamicState(),
                outlet=ThermodynamicState(),
            ),
            turbine=Turbine(
                fluid=FluidProperties(),
                inlet=ThermodynamicState(),
                outlet=ThermodynamicState(),
            ),
        )

        # The cycle-level answer is unchanged.
        _ = cycle.net_specific_work
        _ = cycle.shaft_power
        _ = cycle.thermal_efficiency

        # The component results are also directly observable.
        _ = cycle.compressor.specific_work
        _ = cycle.combustor.heat_addition
        _ = cycle.turbine.specific_work
        _ = cycle.turbine.outlet.temperature

        return cycle


if __name__ == "__main__":
    print(public_example())

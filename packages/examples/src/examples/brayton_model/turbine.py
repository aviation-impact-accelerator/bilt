from __future__ import annotations

import bilt

from .interfaces import (
    FluidProperties,
    ThermodynamicState,
    Turbine,
)


@bilt.core.transform
def outlet_temperature(
    fluid: FluidProperties,
    inlet: ThermodynamicState,
    outlet: ThermodynamicState,
    isentropic_efficiency: float,
) -> float:
    exponent = (
        fluid.heat_capacity_ratio - 1.0
    ) / fluid.heat_capacity_ratio
    pressure_ratio = outlet.pressure / inlet.pressure
    ideal_temperature = (
        inlet.temperature
        * pressure_ratio**exponent
    )

    return (
        inlet.temperature
        - isentropic_efficiency
        * (inlet.temperature - ideal_temperature)
    )


@bilt.core.transform
def specific_work(
    fluid: FluidProperties,
    inlet: ThermodynamicState,
    outlet: ThermodynamicState,
) -> float:
    return (
        fluid.specific_heat_capacity
        * (inlet.temperature - outlet.temperature)
    )


def isentropic_turbine(
    *,
    efficiency: float,
) -> bilt.core.Implementation:
    return bilt.core.Implementation(
        Turbine,
        {
            "outlet.temperature": outlet_temperature,
            "specific_work": specific_work,
        },
        internal_values={
            "isentropic_efficiency": efficiency,
        },
    )

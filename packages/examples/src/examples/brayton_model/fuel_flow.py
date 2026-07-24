from __future__ import annotations

import bilt

from .interfaces import FuelFlow, FuelProperties


@bilt.core.transform
def heat_release(
    fuel: FuelProperties,
    mass_flow_rate: float,
) -> float:
    return (
        fuel.combustion_efficiency
        * fuel.lower_heating_value
        * mass_flow_rate
    )


simple_fuel_flow = bilt.core.Implementation(
    FuelFlow,
    {
        "heat_release": heat_release,
    },
)

from math import exp
from typing import NamedTuple


def heat_capacity_rate(
    mass_flow_rate: float,
    specific_heat_capacity: float,
) -> float:
    return mass_flow_rate * specific_heat_capacity


def velocity(
    mass_flow_rate: float,
    density: float,
    flow_area: float,
) -> float:
    return mass_flow_rate / (density * flow_area)


def local_effectiveness(
    thermal_conductance_per_element: float,
    unmixed_heat_capacity_rate: float,
) -> float:
    return 1.0 - exp(-thermal_conductance_per_element / unmixed_heat_capacity_rate)


def heat_transfer_rate_mixed_to_unmixed(
    mixed_inlet_temperature: float,
    unmixed_inlet_temperature: float,
    mixed_heat_capacity_rate: float,
    unmixed_heat_capacity_rate: float,
    effectiveness: float,
) -> float:
    return (
        effectiveness
        * unmixed_heat_capacity_rate
        * (mixed_inlet_temperature - unmixed_inlet_temperature)
        / (1.0 + effectiveness * unmixed_heat_capacity_rate / (2.0 * mixed_heat_capacity_rate))
    )


def mixed_outlet_temperature(
    mixed_inlet_temperature: float,
    heat_transfer_rate_mixed_to_unmixed: float,
    mixed_heat_capacity_rate: float,
) -> float:
    return mixed_inlet_temperature - heat_transfer_rate_mixed_to_unmixed / mixed_heat_capacity_rate


def unmixed_outlet_temperature(
    unmixed_inlet_temperature: float,
    heat_transfer_rate_mixed_to_unmixed: float,
    unmixed_heat_capacity_rate: float,
) -> float:
    return unmixed_inlet_temperature + heat_transfer_rate_mixed_to_unmixed / unmixed_heat_capacity_rate


class MixedUnmixedElementResult(NamedTuple):
    mixed_outlet_temperature: float
    unmixed_outlet_temperature: float
    heat_transfer_rate_mixed_to_unmixed: float


def mixed_unmixed_element_temperatures(
    mixed_inlet_temperature: float,
    unmixed_inlet_temperature: float,
    mixed_heat_capacity_rate: float,
    unmixed_heat_capacity_rate: float,
    thermal_conductance_per_element: float,
) -> MixedUnmixedElementResult:
    effectiveness = local_effectiveness(
        thermal_conductance_per_element=thermal_conductance_per_element,
        unmixed_heat_capacity_rate=unmixed_heat_capacity_rate,
    )

    heat_transfer_rate = heat_transfer_rate_mixed_to_unmixed(
        mixed_inlet_temperature=mixed_inlet_temperature,
        unmixed_inlet_temperature=unmixed_inlet_temperature,
        mixed_heat_capacity_rate=mixed_heat_capacity_rate,
        unmixed_heat_capacity_rate=unmixed_heat_capacity_rate,
        effectiveness=effectiveness,
    )

    return MixedUnmixedElementResult(
        mixed_outlet_temperature=mixed_outlet_temperature(
            mixed_inlet_temperature=mixed_inlet_temperature,
            heat_transfer_rate_mixed_to_unmixed=heat_transfer_rate,
            mixed_heat_capacity_rate=mixed_heat_capacity_rate,
        ),
        unmixed_outlet_temperature=unmixed_outlet_temperature(
            unmixed_inlet_temperature=unmixed_inlet_temperature,
            heat_transfer_rate_mixed_to_unmixed=heat_transfer_rate,
            unmixed_heat_capacity_rate=unmixed_heat_capacity_rate,
        ),
        heat_transfer_rate_mixed_to_unmixed=heat_transfer_rate,
    )


class TubeOuterElementResult(NamedTuple):
    tube_outlet_temperature: float
    outer_outlet_temperature: float
    heat_transfer_rate_tube_to_outer: float


def tube_outer_element_temperatures(
    tube_inlet_temperature: float,
    outer_inlet_temperature: float,
    tube_heat_capacity_rate: float,
    outer_heat_capacity_rate_per_element: float,
    thermal_conductance_per_element: float,
) -> TubeOuterElementResult:
    element_result = mixed_unmixed_element_temperatures(
        mixed_inlet_temperature=tube_inlet_temperature,
        unmixed_inlet_temperature=outer_inlet_temperature,
        mixed_heat_capacity_rate=tube_heat_capacity_rate,
        unmixed_heat_capacity_rate=outer_heat_capacity_rate_per_element,
        thermal_conductance_per_element=thermal_conductance_per_element,
    )

    return TubeOuterElementResult(
        tube_outlet_temperature=element_result.mixed_outlet_temperature,
        outer_outlet_temperature=element_result.unmixed_outlet_temperature,
        heat_transfer_rate_tube_to_outer=element_result.heat_transfer_rate_mixed_to_unmixed,
    )

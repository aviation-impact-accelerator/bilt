from math import exp
from typing import NamedTuple


def heat_capacity_rate(
    mdot: float,
    specific_heat_capacity: float,
) -> float:
    return mdot * specific_heat_capacity


def velocity(
    mdot: float,
    density: float,
    tube_flow_area: float,
) -> float:
    return mdot / (density * tube_flow_area)


def local_effectiveness(
    thermal_conductance_element: float,
    heat_capacity_rate_unmixed: float,
) -> float:
    return 1.0 - exp(-thermal_conductance_element / heat_capacity_rate_unmixed)


def q_element_mixed_unmixed(
    temp_mixed_in: float,
    temp_unmixed_in: float,
    heat_capacity_rate_mixed: float,
    heat_capacity_rate_unmixed: float,
    local_effectiveness: float,
) -> float:
    return (
        local_effectiveness
        * heat_capacity_rate_unmixed
        * (temp_mixed_in - temp_unmixed_in)
        / (1.0 + local_effectiveness * heat_capacity_rate_unmixed / (2.0 * heat_capacity_rate_mixed))
    )


def temp_mixed_out(
    temp_mixed_in: float,
    q_element_mixed_unmixed: float,
    heat_capacity_rate_mixed: float,
) -> float:
    return temp_mixed_in - q_element_mixed_unmixed / heat_capacity_rate_mixed


def temp_unmixed_out(
    temp_unmixed_in: float,
    q_element_mixed_unmixed: float,
    heat_capacity_rate_unmixed: float,
) -> float:
    return temp_unmixed_in + q_element_mixed_unmixed / heat_capacity_rate_unmixed


class ElementTemperaturesMixedUnmixed(NamedTuple):
    temp_mixed_out: float
    temp_unmixed_out: float
    q_element_mixed_unmixed: float


def element_temperatures_mixed_unmixed(
    temp_mixed_in: float,
    temp_unmixed_in: float,
    heat_capacity_rate_mixed: float,
    heat_capacity_rate_unmixed: float,
    thermal_conductance_element: float,
) -> ElementTemperaturesMixedUnmixed:
    local_effectiveness_value = local_effectiveness(
        thermal_conductance_element=thermal_conductance_element,
        heat_capacity_rate_unmixed=heat_capacity_rate_unmixed,
    )

    q_element_mixed_unmixed_value = q_element_mixed_unmixed(
        temp_mixed_in=temp_mixed_in,
        temp_unmixed_in=temp_unmixed_in,
        heat_capacity_rate_mixed=heat_capacity_rate_mixed,
        heat_capacity_rate_unmixed=heat_capacity_rate_unmixed,
        local_effectiveness=local_effectiveness_value,
    )

    return ElementTemperaturesMixedUnmixed(
        temp_mixed_out=temp_mixed_out(
            temp_mixed_in=temp_mixed_in,
            q_element_mixed_unmixed=q_element_mixed_unmixed_value,
            heat_capacity_rate_mixed=heat_capacity_rate_mixed,
        ),
        temp_unmixed_out=temp_unmixed_out(
            temp_unmixed_in=temp_unmixed_in,
            q_element_mixed_unmixed=q_element_mixed_unmixed_value,
            heat_capacity_rate_unmixed=heat_capacity_rate_unmixed,
        ),
        q_element_mixed_unmixed=q_element_mixed_unmixed_value,
    )


class TubeOuterElementTemperatures(NamedTuple):
    T_tube_out: float
    T_outer_out: float
    q_tube_to_outer: float


def tube_outer_element_temperatures(
    temp_tube_in: float,
    temp_outer_in: float,
    heat_capacity_rate_tube: float,
    heat_capacity_rate_outer_element: float,
    thermal_conductance_element: float,
) -> TubeOuterElementTemperatures:
    element_temperatures_mixed_unmixed_value = element_temperatures_mixed_unmixed(
        temp_mixed_in=temp_tube_in,
        temp_unmixed_in=temp_outer_in,
        heat_capacity_rate_mixed=heat_capacity_rate_tube,
        heat_capacity_rate_unmixed=heat_capacity_rate_outer_element,
        thermal_conductance_element=thermal_conductance_element,
    )

    return TubeOuterElementTemperatures(
        T_tube_out=element_temperatures_mixed_unmixed_value.temp_mixed_out,
        T_outer_out=element_temperatures_mixed_unmixed_value.temp_unmixed_out,
        q_tube_to_outer=element_temperatures_mixed_unmixed_value.q_element_mixed_unmixed,
    )


class ElementResult(NamedTuple):
    T_tube_in: float
    T_tube_out: float
    T_outer_in: float
    T_outer_out: float
    q_tube_to_outer: float


class TubeMarchResult(NamedTuple):
    element_results: tuple[ElementResult, ...]
    T_tube_out: float
    q_total_tube_to_outer: float
    T_outer_out_values: tuple[float, ...]

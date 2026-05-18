from typing import TYPE_CHECKING, NamedTuple

from examples.hx._physics import tube_outer_element_temperatures

if TYPE_CHECKING:
    from collections.abc import Iterable


class TubeOuterElementMarchResult(NamedTuple):
    tube_inlet_temperature: float
    tube_outlet_temperature: float
    outer_inlet_temperature: float
    outer_outlet_temperature: float
    heat_transfer_rate_tube_to_outer: float


class TubeMarchResult(NamedTuple):
    element_results: tuple[TubeOuterElementMarchResult, ...]
    tube_outlet_temperature: float
    total_heat_transfer_rate_tube_to_outer: float
    outer_outlet_temperature_values: tuple[float, ...]


def march_single_tube(
    tube_inlet_temperature: float,
    outer_inlet_temperature_values: Iterable[float],
    tube_heat_capacity_rate: float,
    outer_heat_capacity_rate_values: Iterable[float],
    thermal_conductance_values: Iterable[float],
) -> TubeMarchResult:
    tube_temperature = tube_inlet_temperature
    element_results: list[TubeOuterElementMarchResult] = []

    for (
        outer_inlet_temperature,
        outer_heat_capacity_rate,
        thermal_conductance,
    ) in zip(
        outer_inlet_temperature_values,
        outer_heat_capacity_rate_values,
        thermal_conductance_values,
        strict=True,
    ):
        element_result = tube_outer_element_temperatures(
            tube_inlet_temperature=tube_temperature,
            outer_inlet_temperature=outer_inlet_temperature,
            tube_heat_capacity_rate=tube_heat_capacity_rate,
            outer_heat_capacity_rate_per_element=outer_heat_capacity_rate,
            thermal_conductance_per_element=thermal_conductance,
        )

        element_results.append(
            TubeOuterElementMarchResult(
                tube_inlet_temperature=tube_temperature,
                tube_outlet_temperature=element_result.tube_outlet_temperature,
                outer_inlet_temperature=outer_inlet_temperature,
                outer_outlet_temperature=element_result.outer_outlet_temperature,
                heat_transfer_rate_tube_to_outer=element_result.heat_transfer_rate_tube_to_outer,
            )
        )

        tube_temperature = element_result.tube_outlet_temperature

    total_heat_transfer_rate_tube_to_outer = sum(
        element_result.heat_transfer_rate_tube_to_outer for element_result in element_results
    )

    outer_outlet_temperature_values = tuple(
        element_result.outer_outlet_temperature for element_result in element_results
    )

    return TubeMarchResult(
        element_results=tuple(element_results),
        tube_outlet_temperature=tube_temperature,
        total_heat_transfer_rate_tube_to_outer=total_heat_transfer_rate_tube_to_outer,
        outer_outlet_temperature_values=outer_outlet_temperature_values,
    )

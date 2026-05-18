from typing import TYPE_CHECKING, NamedTuple

from examples.hx._physics import tube_outer_element_temperatures

if TYPE_CHECKING:
    from collections.abc import Iterable


class ElementResult(NamedTuple):
    temp_tube_in: float
    temp_tube_out: float
    temp_outer_in: float
    temp_outer_out: float
    q_tube_to_outer: float


class TubeMarchResult(NamedTuple):
    element_results: tuple[ElementResult, ...]
    temp_tube_out: float
    q_total_tube_to_outer: float
    temp_outer_out_values: tuple[float, ...]


def march_single_tube(
    temp_tube_in: float,
    temp_outer_in_elements: Iterable[float],
    heat_capacity_rate_tube: float,
    heat_capacity_rate_outer_elements: Iterable[float],
    thermal_conductance_elements: Iterable[float],
) -> TubeMarchResult:
    temp_tube = temp_tube_in
    element_results: list[ElementResult] = []

    for temp_outer_in, heat_capacity_rate_outer_element, thermal_conductance_element_value in zip(
        temp_outer_in_elements,
        heat_capacity_rate_outer_elements,
        thermal_conductance_elements,
        strict=True,
    ):
        tube_outer_element_temperatures_value = tube_outer_element_temperatures(
            temp_tube_in=temp_tube,
            temp_outer_in=temp_outer_in,
            heat_capacity_rate_tube=heat_capacity_rate_tube,
            heat_capacity_rate_outer_element=heat_capacity_rate_outer_element,
            thermal_conductance_element=thermal_conductance_element_value,
        )

        element_results.append(
            ElementResult(
                temp_tube_in=temp_tube,
                temp_tube_out=tube_outer_element_temperatures_value.temp_tube_out,
                temp_outer_in=temp_outer_in,
                temp_outer_out=tube_outer_element_temperatures_value.temp_outer_out,
                q_tube_to_outer=tube_outer_element_temperatures_value.q_tube_to_outer,
            )
        )

        temp_tube = tube_outer_element_temperatures_value.temp_tube_out

    q_total_tube_to_outer = sum(element_result.q_tube_to_outer for element_result in element_results)

    temp_outer_out_values = tuple(element_result.temp_outer_out for element_result in element_results)

    return TubeMarchResult(
        element_results=tuple(element_results),
        temp_tube_out=temp_tube,
        q_total_tube_to_outer=q_total_tube_to_outer,
        temp_outer_out_values=temp_outer_out_values,
    )

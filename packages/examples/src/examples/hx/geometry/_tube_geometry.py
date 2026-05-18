from math import pi


def tube_inner_radius(
    tube_outer_radius: float,
    tube_wall_thickness: float,
) -> float:
    return tube_outer_radius - tube_wall_thickness


def tube_inner_diameter(tube_inner_radius: float) -> float:
    return 2.0 * tube_inner_radius


def tube_flow_area(tube_inner_radius: float) -> float:
    return pi * tube_inner_radius**2


def element_length(
    tube_length: float,
    n_elements: int,
) -> float:
    return tube_length / n_elements


def tube_outer_area_element(
    tube_outer_radius: float,
    element_length: float,
) -> float:
    return 2.0 * pi * tube_outer_radius * element_length

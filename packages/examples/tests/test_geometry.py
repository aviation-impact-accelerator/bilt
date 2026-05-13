from math import pi

import pytest
from examples.hx.geometry._tube_geometry import (
    element_length,
    tube_flow_area,
    tube_inner_diameter,
    tube_inner_radius,
    tube_outer_area_element,
)


def test_tube_inner_radius() -> None:
    result = tube_inner_radius(
        tube_outer_radius=0.01,
        tube_wall_thickness=0.001,
    )

    assert result == pytest.approx(0.009)


def test_tube_inner_diameter() -> None:
    result = tube_inner_diameter(tube_inner_radius=0.009)

    assert result == pytest.approx(0.018)


def test_tube_flow_area() -> None:
    result = tube_flow_area(tube_inner_radius=0.009)

    expected = pi * 0.009**2

    assert result == pytest.approx(expected)


def test_element_length() -> None:
    result = element_length(
        tube_length=1.0,
        n_elements=10,
    )

    assert result == pytest.approx(0.1)


def test_tube_outer_area_element() -> None:
    result = tube_outer_area_element(
        tube_outer_radius=0.01,
        element_length=0.1,
    )

    expected = 2.0 * pi * 0.01 * 0.1

    assert result == pytest.approx(expected)

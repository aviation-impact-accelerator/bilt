import numpy as np
import pytest

from bilt.core._domain_registry import (
    DomainRegistry,
    DomainRegistryError,
)
from bilt.core._tensors._tensors import tensor
from bilt.symbolic._dimensions import coordinate_axis, dim


def test_registry_binds_domain_values() -> None:
    location = dim("location")
    time = dim("time")

    value = tensor(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ],
        over=(location, time),
    )

    registry = DomainRegistry()
    registry.bind_all((1.0, value))

    size_location = 2
    size_time = 3

    assert registry.size(location) == size_location
    assert registry.size(time) == size_time


def test_registry_rejects_inconsistent_dimension_sizes() -> None:
    first_time = dim("time")
    second_time = dim("time")

    registry = DomainRegistry()
    registry.bind(tensor([1.0, 2.0], over=first_time))

    with pytest.raises(
        DomainRegistryError,
        match="inconsistent sizes",
    ):
        registry.bind(
            tensor([1.0, 2.0, 3.0], over=second_time),
        )


def test_registry_binds_coordinate_axis() -> None:
    location = dim("location")
    axis = coordinate_axis(
        location,
        ("London", "Manchester"),
    )

    registry = DomainRegistry()
    registry.bind(axis)

    size_location = 2
    position_location = 1

    assert registry.size(location) == size_location
    assert registry.axis(location) == axis
    assert registry.position(location, "Manchester") == position_location


def test_registry_upgrades_size_binding_to_axis() -> None:
    location = dim("location")
    axis = coordinate_axis(
        location,
        ("London", "Manchester"),
    )

    registry = DomainRegistry()
    registry.bind_value(tensor([1.0, 2.0], over=location))
    registry.bind_axis(axis)

    assert registry.axis(location) == axis


def test_registry_accepts_equivalent_coordinate_axis() -> None:
    location = dim("location")

    registry = DomainRegistry()
    registry.bind_axis(
        coordinate_axis(
            location,
            ("London", "Manchester"),
        )
    )
    registry.bind_axis(
        coordinate_axis(
            location,
            ("London", "Manchester"),
        )
    )

    assert registry.position(location, "Manchester") == 1


def test_registry_rejects_axis_with_inconsistent_size() -> None:
    location = dim("location")

    registry = DomainRegistry()
    registry.bind_value(tensor([1.0, 2.0, 3.0], over=location))

    with pytest.raises(
        DomainRegistryError,
        match="inconsistent sizes",
    ):
        registry.bind_axis(
            coordinate_axis(
                location,
                ("London", "Manchester"),
            )
        )


def test_registry_rejects_conflicting_coordinates() -> None:
    location = dim("location")

    registry = DomainRegistry()
    registry.bind(
        coordinate_axis(
            location,
            ("London", "Manchester"),
        )
    )

    with pytest.raises(
        DomainRegistryError,
        match="conflicting coordinate definitions",
    ):
        registry.bind(
            coordinate_axis(
                location,
                ("London", "Birmingham"),
            )
        )


COORDS = np.linspace(0.0, 1.0, 21)
EXPECTED = [
    0.0,
    0.05,
    0.1,
    0.15,
    0.2,
    0.25,
    0.3,
    0.35,
    0.4,
    0.45,
    0.5,
    0.55,
    0.6,
    0.65,
    0.7,
    0.75,
    0.8,
    0.85,
    0.9,
    0.95,
    1.0,
]


@pytest.mark.parametrize("index", range(len(COORDS)))  # pyright: ignore[reportDeprecated]
def test_domain_registry_copes_with_float_coordinates(index: int) -> None:
    time = dim("time")

    registry = DomainRegistry()
    registry.bind_axis(coordinate_axis(time, COORDS))

    expected_value = EXPECTED[index]

    assert registry.position(time, expected_value) == index

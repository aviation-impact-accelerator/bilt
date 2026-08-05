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

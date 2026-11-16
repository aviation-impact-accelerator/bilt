import numpy as np
import pytest

from bilt.core._domain_registry import DomainRegistry
from bilt.core._tensors._tensors import tensor
from bilt.errors import BiltDomainRegistryError
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
        BiltDomainRegistryError,
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
        BiltDomainRegistryError,
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
        BiltDomainRegistryError,
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


def test_concat_derives_size_without_freezing_source_bindings() -> None:
    past, future, complete = dim("past"), dim("future"), dim("complete")
    registry = DomainRegistry()
    registry.bind(tensor([1.0, 2.0], over=past))
    registry.bind(tensor([3.0, 4.0, 5.0], over=future))
    assert registry.dimension_order().dims == (past, future)
    registry.bind_concat((past, future), complete)
    assert registry.size(complete) == 5
    assert registry.has_derived_dimensions


def test_concat_derives_coordinates_and_rejects_partial_labels() -> None:
    past, future, complete = dim("past"), dim("future"), dim("complete")
    registry = DomainRegistry()
    registry.bind_axis(coordinate_axis(past, (2023, 2024)))
    registry.bind_axis(coordinate_axis(future, (2025, 2026)))
    registry.bind_concat((past, future), complete)
    assert registry.axis(complete).coordinates == (2023, 2024, 2025, 2026)

    partial = DomainRegistry()
    partial.bind_axis(coordinate_axis(past, (2023,)))
    partial.bind(tensor([1.0], over=future))
    with pytest.raises(BiltDomainRegistryError, match="labelled and unlabelled"):
        partial.bind_concat((past, future), complete)


def test_derived_registry_clone_isolated_from_original() -> None:
    source, extra, combined = dim("source"), dim("extra"), dim("combined")
    original = DomainRegistry()
    original.bind(tensor([1.0, 2.0], over=source))
    clone = original.clone()
    clone.bind(tensor([3.0], over=extra))
    clone.bind_concat((source, extra), combined)
    assert clone.size(combined) == 3
    assert not original.has_derived_dimensions
    with pytest.raises(BiltDomainRegistryError, match="not bound"):
        original.size(combined)


def test_concat_rejects_conflicting_existing_target() -> None:
    past, future, complete = dim("past"), dim("future"), dim("complete")
    registry = DomainRegistry()
    registry.bind(tensor([1.0, 2.0], over=past))
    registry.bind(tensor([3.0], over=future))
    registry.bind(tensor([10.0, 11.0], over=complete))
    with pytest.raises(BiltDomainRegistryError, match="inconsistent sizes"):
        registry.bind_concat((past, future), complete)

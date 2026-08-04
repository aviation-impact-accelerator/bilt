import datetime

from bilt.symbolic._dimensions import coordinate_axis, dim


def test_dimensions_with_same_name_are_equal() -> None:
    first = dim("time")
    second = dim("time")

    assert first is not second
    assert first == second


def test_dimensions_with_different_names_are_not_equal() -> None:
    assert dim("time") != dim("location")


def test_coordinate_axis_binds_coordinates_to_dimension() -> None:
    time = dim("time")
    coordinates = (
        datetime.date(2026, 1, 1),
        datetime.date(2026, 2, 1),
        datetime.date(2026, 3, 1),
    )

    axis = coordinate_axis(time, coordinates)

    assert axis.dimension is time
    assert axis.coordinates == coordinates
    assert len(axis) == 3
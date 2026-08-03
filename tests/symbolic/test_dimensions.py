from bilt.symbolic._dimensions import coordinate_axis, dim


def test_dimension_uses_object_identity() -> None:
    first = dim("time")
    second = dim("time")

    assert first is not second
    assert first != second


def test_same_dimension_can_be_shared() -> None:
    time = dim("time")

    first_reference = time
    second_reference = time

    assert first_reference is second_reference


def test_coordinate_axis_binds_coordinates_to_dimension() -> None:
    time = dim("time")

    axis = coordinate_axis(
        time,
        ["2026-01-01", "2026-02-01", "2026-03-01"],
    )

    assert axis.dimension is time
    assert axis.coordinates == (
        "2026-01-01",
        "2026-02-01",
        "2026-03-01",
    )
    length = 3
    assert len(axis) == length

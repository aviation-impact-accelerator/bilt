import pytest

from bilt.symbolic._dimensions import dim
from bilt.symbolic._domains import Domain, as_domain, domain


def test_scalar_domain() -> None:
    scalar = Domain.scalar()

    assert scalar.dims == ()
    assert scalar.is_scalar
    assert str(scalar) == "scalar"


def test_domain_preserves_dimension_order() -> None:
    time = dim("time")
    location = dim("location")

    result = domain(location, time)

    assert result.dims == (location, time)
    assert str(result) == "[location, time]"


def test_as_domain_converts_supported_values() -> None:
    time = dim("time")
    existing = domain(time)

    assert as_domain(time) == existing
    assert as_domain((time,)) == existing
    assert as_domain([time]) == existing
    assert as_domain(existing) is existing


def test_domain_rejects_repeated_dimension() -> None:
    time = dim("time")

    with pytest.raises(
        ValueError,
        match="cannot contain the same dimension more than once",
    ):
        domain(time, time)


def test_domain_rejects_dimensions_with_same_name() -> None:
    first_time = dim("time")
    second_time = dim("time")

    with pytest.raises(
        ValueError,
        match="cannot contain the same dimension more than once",
    ):
        domain(first_time, second_time)

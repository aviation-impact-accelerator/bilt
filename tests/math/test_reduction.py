from bilt.symbolic._atoms import DimensionType, dimension
from bilt.symbolic._dimensions import Dim


def test_dimension_atom_preserves_dimension() -> None:
    time = Dim("time")

    result = dimension(time)

    assert result.atom_type is DimensionType
    assert result.value == time


def test_dimension_atom_is_hash_consed() -> None:
    time = Dim("time")

    assert dimension(time) is dimension(time)

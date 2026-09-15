from bilt.symbolic._atoms import DimensionType, dimension
from bilt.symbolic._dimensions import Dim


def test_dimension_atom_preserves_dim_identity() -> None:
    time = Dim("time")

    result = dimension(time)

    assert result.atom_type is DimensionType
    assert result.value is time


def test_dimension_atom_is_hash_consed() -> None:
    time = Dim("time")

    assert dimension(time) is dimension(time)

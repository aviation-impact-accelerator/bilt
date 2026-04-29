from bilt.units.units import multiply


def test_bilt_units_multiply() -> None:
    result = 6
    assert multiply(2, 3) == result

import pytest

from bilt.units import Prefix


class TestPrefix:
    @staticmethod
    @pytest.mark.parametrize("value", (-17, -4, 0, 5, None))
    def test_instantiation_with_invalid_prefix_value(value: int | None) -> None:
        with pytest.raises(ValueError, match=f"{value} is not a valid Prefix"):
            _ = Prefix(value)  # type: ignore[arg-type]

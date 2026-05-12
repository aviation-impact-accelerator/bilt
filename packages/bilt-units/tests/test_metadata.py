from __future__ import annotations

import pytest
from bilt.units import Metadata
from bilt.units._metadata import MetadataError


class TestMetadata:
    @staticmethod
    def test_empty_metadata_is_invalid() -> None:
        with pytest.raises(MetadataError, match="Metadata must be a non-empty value"):
            _ = Metadata("")

    @staticmethod
    @pytest.mark.parametrize(
        ("left", "right", "is_less_than"),
        (
            (Metadata("co2"), Metadata("nox"), True),
            (Metadata("co2"), None, False),
            (None, Metadata("co2"), True),
        ),
    )
    def test_comparison(left: Metadata | None, right: Metadata | None, is_less_than: bool) -> None:
        assert (left < right) is is_less_than  # type: ignore[operator]

    @staticmethod
    def test_equality() -> None:
        assert Metadata("something") == Metadata("something")

    @staticmethod
    @pytest.mark.parametrize(
        ("metadata", "expected_str", "expected_repr"),
        (
            (Metadata("Thermal"), "Thermal", "Metadata(value='Thermal')"),
            (Metadata("co2"), "co2", "Metadata(value='co2')"),
        ),
    )
    def test_representation(metadata: Metadata, expected_str: str, expected_repr: str) -> None:
        assert repr(metadata) == expected_repr
        assert str(metadata) == expected_str

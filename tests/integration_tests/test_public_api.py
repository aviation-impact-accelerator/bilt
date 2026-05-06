"""Smoke tests for the public bilt API."""

import bilt
import bilt.core


def test_core_public_api_is_reexported_from_bilt() -> None:
    """Every public core export should also be available from bilt."""
    for name in bilt.core.__all__:
        assert hasattr(bilt, name), f"bilt is missing public core export: {name}"

        top_level_object = getattr(bilt, name)
        core_object = getattr(bilt.core, name)

        assert top_level_object is core_object, f"bilt.{name} is not the same object as bilt.core.{name}"


def test_bilt_all_names_exist() -> None:
    """Every name in bilt.__all__ should exist on bilt."""
    for name in bilt.__all__:
        assert hasattr(bilt, name), f"bilt.__all__ contains missing name: {name}"


def test_bilt_all_includes_core_all() -> None:
    """The top-level bilt API should include all public core exports."""
    missing = set(bilt.core.__all__) - set(bilt.__all__)

    assert not missing, f"bilt.__all__ is missing core exports: {sorted(missing)}"

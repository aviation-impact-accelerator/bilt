"""Errors for units and dimensional analysis."""

__all__ = [
    "BiltDimensionalityError",
    "BiltMetadataError",
    "BiltUnitError",
    "BiltUnitsError",
]

from bilt._base_error import BiltError


class BiltUnitsError(BiltError):
    """Base error for unit and dimensionality failures."""


class BiltUnitError(BiltUnitsError):
    """Raised when a unit specification or conversion is invalid."""


class BiltDimensionalityError(BiltUnitsError):
    """Raised when a dimensionality check fails."""


class BiltMetadataError(BiltUnitsError):
    """Raised when unit or dimension metadata is invalid."""

"""Errors for Bilt model-checker configuration."""

__all__ = ["BiltCheckConfigurationError", "BiltCheckError"]

from bilt._base_error import BiltError


class BiltCheckError(BiltError):
    """Base error for model checking."""


class BiltCheckConfigurationError(BiltCheckError, ValueError):
    """Invalid configuration for Bilt's model checker."""

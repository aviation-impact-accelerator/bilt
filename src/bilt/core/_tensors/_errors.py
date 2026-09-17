"""Errors for tensor domain validation and symbolic lowering."""

__all__ = ["BiltSymbolicLoweringError", "BiltTensorDomainError"]

from bilt.core._errors import BiltEvaluationError


class BiltTensorDomainError(BiltEvaluationError, ValueError):
    """Raised when concrete tensor values do not match their declared domain."""


class BiltSymbolicLoweringError(BiltEvaluationError, ValueError):
    """Raised when lowering a symbolic expression to a concrete form fails."""

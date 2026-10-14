from bilt.core._errors import BiltAnalysisError


class BiltUnderconstrainedError(BiltAnalysisError):
    """The scoped problem has insufficient constraints."""


class BiltOverconstrainedError(BiltAnalysisError):
    """The scoped problem has competing or surplus constraints."""

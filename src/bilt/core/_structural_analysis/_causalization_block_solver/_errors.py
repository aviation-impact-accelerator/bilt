from bilt.core._errors import BiltAnalysisError


class BiltUnderconstrainedError(BiltAnalysisError):
    """The scoped problem has insufficient constraints."""


class BiltOverconstrainedError(BiltAnalysisError):
    """The scoped problem has competing or surplus constraints."""


class BiltExecutionPlanError(BiltAnalysisError):
    """A structurally validated request has no supported execution plan."""

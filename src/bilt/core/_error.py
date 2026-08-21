__all__ = ["BiltCoreError", "BiltNameError"]


class BiltCoreError(Exception):
    pass


class EquationReturnError(BiltCoreError):
    """Raised when an equation function does not return a Relation."""


class BiltNameError(BiltCoreError):
    """Raised when a declared name collides with the separator reserved for flattened port names."""

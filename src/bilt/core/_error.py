__all__ = ["BiltCoreError"]


class BiltCoreError(Exception):
    pass


class EquationReturnError(BiltCoreError):
    """Raised when an equation function does not return a Relation."""

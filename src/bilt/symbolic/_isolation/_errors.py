__all__ = [
    "IsolationError",
    "TargetNotFoundError",
    "TargetOnBothSidesError",
    "UnsupportedIsolationError",
]


class IsolationError(Exception):
    pass


class TargetNotFoundError(IsolationError):
    pass


class TargetOnBothSidesError(IsolationError):
    pass


class UnsupportedIsolationError(IsolationError):
    pass

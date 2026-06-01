__all__ = ["AtomError", "AtomMutationError", "BiltSymbolicError"]


class BiltSymbolicError(Exception):
    """Base error for the bilt-symbolic library."""


class AtomError(BiltSymbolicError):
    """Base error for atom construction and hash-consing failures."""


class AtomMutationError(AtomError, AttributeError):
    """Raised when attempting to mutate an immutable Atom."""

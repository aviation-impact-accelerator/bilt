__all__ = ["AtomError", "AtomHashError", "AtomMutationError", "BiltSymbolicError"]


class BiltSymbolicError(Exception):
    """Base error for the bilt-symbolic library."""


class AtomError(BiltSymbolicError):
    """Base error for atom construction and hash-consing failures."""


class AtomHashError(AtomError):
    """Raised when an Atom value is not suitable for hash-consing."""


class AtomMutationError(AtomError, AttributeError):
    """Raised when attempting to mutate an immutable Atom."""

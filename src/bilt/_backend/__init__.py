"""Interface for methods to implement by the JAX backend."""

__all__ = ("Backend", "BackendCase")

from bilt._backend._case import BackendCase
from bilt._backend._protocol import Backend

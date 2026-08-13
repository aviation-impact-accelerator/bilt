"""Interface classes for using backends in bilt.core."""

__all__ = ("Backend", "BackendCallable", "BackendCase")

from bilt._backend._base_backend import Backend
from bilt._backend._callable import BackendCallable
from bilt._backend._case import BackendCase

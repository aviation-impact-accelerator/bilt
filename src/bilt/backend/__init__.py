"""Interface classes for using backends in bilt.core."""

__all__ = ("Backend", "BackendCallable", "BackendCase")

from bilt.backend._base_backend import Backend
from bilt.backend._callable import BackendCallable
from bilt.backend._case import BackendCase

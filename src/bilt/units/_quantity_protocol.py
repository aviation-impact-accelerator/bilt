"""Shared, representation-independent typing for quantities.

This intentionally does not import Tensor (or Unit) at runtime. A quantity
can carry a scalar or tensor magnitude, but the two concrete representations
do not need to inherit from a common runtime implementation.
"""

import typing

if typing.TYPE_CHECKING:
    from bilt.units._units import Unit


class QuantityLike[M = object](typing.Protocol):
    """Read-only quantity contract, covariant in the magnitude type.

    The default is `object` for now because it accepts both scalar and
    tensor magnitudes without importing Tensor into the units layer.
    A production tensor quantity and richer arithmetic protocol can follow
    independently.
    """

    @property
    def value(self) -> M:
        """Return the unlabelled scalar or tensor magnitude."""
        ...

    @property
    def unit(self) -> Unit:
        """Return the non-optional physical unit."""
        ...

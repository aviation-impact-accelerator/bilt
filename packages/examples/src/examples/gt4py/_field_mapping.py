"""Map Bilt's named tensor domains onto GT4Py Next field dimensions.

This proof of concept lives in the examples package, not Bilt core.
It supports a strict one-to-one mapping of dimensions; batching, halos and
non-spatial dimensions are out of scope.
"""

from __future__ import annotations

import dataclasses
import typing

import numpy as np

from bilt import Domain, Tensor, tensor
from bilt.symbolic._dimensions import Dim

if typing.TYPE_CHECKING:
    from bilt.units import Unit


class GT4PyField(typing.Protocol):
    """The field method this adapter needs to restore values."""

    def asnumpy(self) -> np.ndarray: ...


@dataclasses.dataclass(frozen=True, slots=True)
class CartesianFieldMapping:
    """Associate Bilt dimensions with ordered GT4Py Next dimensions.

    The axes are ordered as expected by the GT4Py field operator.
    GT4Py dimensions are opaque here to keep the dependency optional.
    """

    axes: tuple[tuple[Dim, object], ...]

    def __post_init__(self) -> None:
        if not self.axes:
            msg = "A GT4Py field mapping needs at least one dimension."
            raise ValueError(msg)

        bilt_dims = tuple(dim for dim, _ in self.axes)
        gt_dims = tuple(dim for _, dim in self.axes)
        if len(set(bilt_dims)) != len(bilt_dims):
            msg = "A Bilt dimension cannot appear twice in a GT4Py field mapping."
            raise ValueError(msg)
        if len(set(gt_dims)) != len(gt_dims):
            msg = "A GT4Py dimension cannot appear twice in a field mapping."
            raise ValueError(msg)

    @property
    def domain(self) -> Domain:
        """Bilt dimensions in GT4Py field-axis order."""
        return Domain(tuple(dim for dim, _ in self.axes))

    def to_field(self, value: Tensor | np.ndarray, *, domain: Domain | None = None) -> GT4PyField:
        """Convert a Bilt Tensor or an array with an explicit Bilt domain.

        The array form supports transforms, whose parameters are lowered to
        raw numerical arrays before their Python implementations are called.
        """
        if isinstance(value, Tensor):
            if domain is not None:
                msg = "A Tensor already carries its domain; do not pass domain=."
                raise ValueError(msg)
            domain = value.domain
            values = value.values
        else:
            if domain is None:
                msg = "Raw arrays require domain= to identify their current axis order."
                raise ValueError(msg)
            values = np.asarray(value)

        self._validate_domain(domain, values.ndim)
        axis_order = tuple(domain.dims.index(dim) for dim in self.domain.dims)
        reordered = values.transpose(axis_order)

        import gt4py.next as gtx  # noqa: PLC0415

        return typing.cast("GT4PyField", gtx.as_field([gt_dim for _, gt_dim in self.axes], reordered))

    def to_array(self, field: GT4PyField, *, domain: Domain) -> np.ndarray:
        """Restore GT4Py values to a requested Bilt axis order."""
        values = np.asarray(field.asnumpy())
        self._validate_domain(domain, values.ndim)
        axis_order = tuple(self.domain.dims.index(dim) for dim in domain.dims)
        return values.transpose(axis_order)

    def to_tensor(self, field: GT4PyField, *, domain: Domain, unit: Unit | None = None) -> Tensor:
        """Restore a GT4Py field as a concrete Bilt Tensor."""
        return tensor(self.to_array(field, domain=domain), domain=domain, unit=unit)

    def to_runtime_field(self, values: np.ndarray) -> GT4PyField:
        """Convert an aligned array inside an active Bilt transform.

        This temporary bridge uses Bilt's private evaluation context until
        a public transform adapter can receive domain metadata explicitly.
        """
        from bilt.core._evaluation_context import current_evaluation  # noqa: PLC0415

        return self.to_field(values, domain=current_evaluation().canonical_order)

    def to_runtime_array(self, field: GT4PyField) -> np.ndarray:
        """Restore GT4Py output to Bilt's evaluation-wide axis order."""
        from bilt.core._evaluation_context import current_evaluation  # noqa: PLC0415

        return self.to_array(field, domain=current_evaluation().canonical_order)

    def _validate_domain(self, domain: Domain, ndim: int) -> None:
        if ndim != len(domain.dims):
            msg = f"Array rank {ndim} does not match Bilt domain {domain}."
            raise ValueError(msg)
        if len(domain.dims) != len(self.axes) or set(domain.dims) != set(self.domain.dims):
            msg = (
                f"Bilt domain {domain} must be a permutation of mapped field dimensions {self.domain}; "
                "additional batch dimensions are not supported yet."
            )
            raise ValueError(msg)

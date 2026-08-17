import dataclasses
import typing

import numpy as np

from bilt.core._evaluation_context import current_evaluation
from bilt.core._tensors import _tensors
from bilt.core._tensors._tensors import Tensor
from bilt.symbolic._domains import Domain

if typing.TYPE_CHECKING:
    from bilt.symbolic._dimensions import Dim


@dataclasses.dataclass(frozen=True)
class CanonicalDomain:
    domain: Domain


def canonicalised(domain: Domain) -> CanonicalDomain:
    """Puts dimensions in canonical order.

    Order determined by the order of dimensions in the current
    evaluation's domain registry.
    """
    canonicalised_dims: list[Dim] = [dim for dim in _get_canonical_domain().domain.dims if dim in domain.dims]

    return CanonicalDomain(Domain(tuple(canonicalised_dims)))


def transpose_and_align_to_canonical(tensor: Tensor) -> Tensor:
    canonical_domain = _get_canonical_domain()
    transposed = _transpose_tensor_to_domain_order(tensor, canonical_domain.domain)
    return _align_tensor_to_domain_order(transposed, canonical_domain.domain)


def _transpose_tensor_to_domain_order(tensor: Tensor, domain: Domain) -> Tensor:
    transposition_axes: list[int] = []
    transposed_dims: list[Dim] = []

    for dim in domain.dims:
        if dim in tensor.domain.dims:
            transposition_axes.append(tensor.domain.dims.index(dim))
            transposed_dims.append(dim)

    transposed_values = tensor.values.transpose(transposition_axes)

    return _tensors.tensor(transposed_values, over=transposed_dims)


def _align_tensor_to_domain_order(tensor: Tensor, order: Domain) -> Tensor:
    """Important: `tensor`'s dims must already be in the same order as `order`."""
    new_values = tensor.values.copy()

    for i, dim in enumerate(order.dims):
        if dim not in tensor.domain.dims:
            new_values = np.expand_dims(new_values, axis=i)

    return Tensor(new_values, Domain(order.dims))


def _get_canonical_domain() -> CanonicalDomain:
    return CanonicalDomain(Domain(tuple(current_evaluation().domains._bindings.keys())))  # noqa: SLF001

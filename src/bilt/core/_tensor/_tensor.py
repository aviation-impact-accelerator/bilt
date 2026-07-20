import collections.abc
import dataclasses

import numpy as np

from bilt.core._tensor._error import TensorDomainError
from bilt.symbolic._dim import Dim
from bilt.symbolic._domains import Domain

__all__ = [
    "BiltTensor",
    "OverArg",
    "align_to",
    "as_bilt_tensor",
    "binary_op",
    "tensor",
]


OverArg = Dim | collections.abc.Iterable[Dim]


@dataclasses.dataclass(frozen=True, slots=True)
class BiltTensor:
    """Concrete values over a named Bilt domain.

    This is intentionally small. It is not an xarray replacement.

    values:
        Backend storage. For now this implementation uses numpy arrays.
    domain:
        The semantic named axes for values.

    _broadcast_axes:
        Private implementation detail used for tensors created by explicit
        scalar lifting or alignment. Axes in this set are allowed to have
        length 1 and broadcast during binary operations. User-supplied
        non-scalar tensors do not get broadcast axes by default, so
        tensor([1.0], over=time) is not silently treated as a scalar.
    """

    values: np.ndarray
    domain: Domain
    _broadcast_axes: frozenset[int] = dataclasses.field(
        default_factory=frozenset[int],
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        values = np.asarray(self.values)
        object.__setattr__(self, "values", values)

        _check_rank(values, self.domain)
        _check_broadcast_axes(values, self._broadcast_axes)

    @classmethod
    def scalar(cls, value: object) -> BiltTensor:
        values = np.asarray(value)

        if values.ndim != 0:
            message = f"Cannot create scalar BiltTensor from rank-{values.ndim} value."
            raise TensorDomainError(message)

        return cls(values, Domain.scalar())

    @property
    def shape(self) -> tuple[int, ...]:
        return self.values.shape

    @property
    def ndim(self) -> int:
        return self.values.ndim

    def align_to(self, target: Domain) -> BiltTensor:
        return align_to(self, target)

    def to_domain(self, target: Domain) -> BiltTensor:
        """Lift or reorder this tensor to a compatible target domain.

        This may add missing dimensions or reorder existing dimensions.
        It must not drop dimensions or rename them.
        """
        return align_to(self, target)

    def over(self, *dims: Dim) -> BiltTensor:
        return self.to_domain(Domain(tuple(dims)))

    def __mul__(self, other: object) -> BiltTensor:
        return binary_op(self, as_bilt_tensor(other), np.multiply)

    def __rmul__(self, other: object) -> BiltTensor:
        return binary_op(as_bilt_tensor(other), self, np.multiply)

    def __add__(self, other: object) -> BiltTensor:
        return binary_op(self, as_bilt_tensor(other), np.add)

    def __radd__(self, other: object) -> BiltTensor:
        return binary_op(as_bilt_tensor(other), self, np.add)

    def __sub__(self, other: object) -> BiltTensor:
        return binary_op(self, as_bilt_tensor(other), np.subtract)

    def __rsub__(self, other: object) -> BiltTensor:
        return binary_op(as_bilt_tensor(other), self, np.subtract)

    def __truediv__(self, other: object) -> BiltTensor:
        return binary_op(self, as_bilt_tensor(other), np.divide)

    def __rtruediv__(self, other: object) -> BiltTensor:
        return binary_op(as_bilt_tensor(other), self, np.divide)

    def __neg__(self) -> BiltTensor:
        return BiltTensor(-self.values, self.domain, self._broadcast_axes)


def tensor(
    values: object,
    *,
    over: OverArg | None = None,
    domain: Domain | None = None,
) -> BiltTensor:
    """Create a BiltTensor from user data.

    Scalars may omit a domain. Non-scalar lists/arrays must provide
    either over= or domain=.

    When a scalar is supplied with a non-scalar domain, it is explicitly
    lifted to that domain using singleton axes.
    """
    if over is not None and domain is not None:
        message = "Pass either over= or domain=, not both."
        raise TensorDomainError(message)

    array = np.asarray(values)

    if domain is None:
        if over is None:
            if array.ndim == 0:
                return BiltTensor.scalar(array)

            message = (
                "Non-scalar values need an explicit domain. Use tensor(values, over=...) or tensor(values, domain=...)."
            )
            raise TensorDomainError(message)

        domain = _domain_from_over(over)

    if array.ndim == 0 and domain.dims:
        return BiltTensor.scalar(array).to_domain(domain)

    return BiltTensor(array, domain)


def as_bilt_tensor(value: object) -> BiltTensor:
    """Coerce scalar values to BiltTensor and reject ambiguous arrays/lists."""
    if isinstance(value, BiltTensor):
        return value

    values = np.asarray(value)

    if values.ndim == 0:
        return BiltTensor.scalar(values)

    message = (
        "Raw non-scalar arrays/lists need an explicit domain. "
        "Use tensor(values, over=...) or "
        "BiltTensor(values, domain=...)."
    )
    raise TensorDomainError(message)


def align_to(tensor: BiltTensor, target: Domain) -> BiltTensor:
    """Align tensor values to a target named domain.

    Allowed:
        time -> time x node          insert singleton node axis
        node x time -> time x node   reorder axes
        scalar -> any target         insert singleton axes

    Not allowed:
        time x node -> time          dropping dimensions
        time -> node                 renaming/reinterpreting dimensions
    """
    source = tensor.domain

    if source == target:
        return tensor

    if not _domain_is_subset_of(source, target):
        message = (
            f"Cannot align domain {source!r} to {target!r}. "
            "The source contains dimensions not present in the target. "
            "Use an explicit rename or reduction operation instead."
        )
        raise TensorDomainError(message)

    values = tensor.values
    source_dims = tuple(source.dims)
    target_dims = tuple(target.dims)

    existing_target_dims = tuple(
        target_dim for target_dim in target_dims if _find_dim_index(target_dim, source_dims) is not None
    )

    permutation = tuple(_require_dim_index(target_dim, source_dims) for target_dim in existing_target_dims)

    if permutation != tuple(range(len(permutation))):
        values = np.transpose(values, axes=permutation)

    broadcast_axes = {
        new_axis
        for new_axis, old_axis in enumerate(permutation)
        if old_axis in tensor._broadcast_axes  # noqa: SLF001
    }

    current_dims = list(existing_target_dims)

    for axis, target_dim in enumerate(target_dims):
        if _find_dim_index(target_dim, current_dims) is None:
            values = np.expand_dims(values, axis=axis)
            current_dims.insert(axis, target_dim)
            broadcast_axes = {
                existing_axis + 1 if existing_axis >= axis else existing_axis for existing_axis in broadcast_axes
            }
            broadcast_axes.add(axis)

    return BiltTensor(values, target, frozenset(broadcast_axes))


def binary_op(
    lhs: BiltTensor,
    rhs: BiltTensor,
    op: collections.abc.Callable[[np.ndarray, np.ndarray], np.ndarray],
) -> BiltTensor:
    """Apply an elementwise binary operation with named-domain broadcasting."""
    result_domain = _domain_broadcast(lhs.domain, rhs.domain)

    lhs_aligned = align_to(lhs, result_domain)
    rhs_aligned = align_to(rhs, result_domain)

    _check_binary_shapes(lhs_aligned, rhs_aligned, result_domain)

    return BiltTensor(
        values=op(lhs_aligned.values, rhs_aligned.values),
        domain=result_domain,
    )


def _domain_from_over(over: OverArg) -> Domain:
    if isinstance(over, Dim):
        return Domain((over,))

    return Domain(tuple(over))


def _check_rank(values: np.ndarray, domain: Domain) -> None:
    domain_rank = len(domain.dims)

    if values.ndim != domain_rank:
        message = f"Value rank {values.ndim} does not match domain rank {domain_rank} for domain {domain!r}."
        raise TensorDomainError(message)


def _check_broadcast_axes(values: np.ndarray, axes: frozenset[int]) -> None:
    invalid_axes = tuple(axis for axis in axes if axis < 0 or axis >= values.ndim)

    if invalid_axes:
        message = f"Broadcast axes {invalid_axes!r} are invalid for shape {values.shape!r}."
        raise TensorDomainError(message)

    non_singleton_axes = tuple(axis for axis in axes if values.shape[axis] != 1)

    if non_singleton_axes:
        message = (
            "Broadcast axes must have length 1; got non-singleton axes "
            f"{non_singleton_axes!r} for shape {values.shape!r}."
        )
        raise TensorDomainError(message)


def _check_binary_shapes(
    lhs: BiltTensor,
    rhs: BiltTensor,
    domain: Domain,
) -> None:
    for axis, dim in enumerate(domain.dims):
        lhs_size = lhs.values.shape[axis]
        rhs_size = rhs.values.shape[axis]

        if lhs_size == rhs_size:
            continue

        if lhs_size == 1 and axis in lhs._broadcast_axes:  # noqa: SLF001
            continue

        if rhs_size == 1 and axis in rhs._broadcast_axes:  # noqa: SLF001
            continue

        message = f"Incompatible sizes for dimension {dim.name!r}: {lhs_size} and {rhs_size}."
        raise TensorDomainError(message)


def _domain_is_subset_of(source: Domain, target: Domain) -> bool:
    return all(_find_dim_index(dim, target.dims) is not None for dim in source.dims)


def _domain_broadcast(lhs: Domain, rhs: Domain) -> Domain:
    """Return the named broadcast domain for two domains.

    Shared dimensions align by semantic name. Missing dimensions are appended
    in right-hand order. The left-hand domain controls the preferred output
    order.
    """
    result = list(lhs.dims)

    for rhs_dim in rhs.dims:
        if _find_dim_index(rhs_dim, result) is None:
            result.append(rhs_dim)

    return Domain(tuple(result))


def _find_dim_index(
    dim: Dim,
    dims: collections.abc.Sequence[Dim],
) -> int | None:
    for index, candidate in enumerate(dims):
        if candidate.name == dim.name:
            return index

    return None


def _require_dim_index(
    dim: Dim,
    dims: collections.abc.Sequence[Dim],
) -> int:
    index = _find_dim_index(dim, dims)

    if index is None:
        message = f"Dimension {dim.name!r} is not present."
        raise TensorDomainError(message)

    return index

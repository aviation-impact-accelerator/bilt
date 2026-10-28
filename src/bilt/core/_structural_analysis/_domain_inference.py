"""Infer a unique least named-axis layout for a supported joint linear block."""

import collections
import dataclasses
import typing

from bilt.core._structural_analysis._linear_block import BiltLinearBlockError
from bilt.symbolic._atoms import DimensionType, IndexOpType, ReductionOpType, SymbolType
from bilt.symbolic._domains import Domain

if typing.TYPE_CHECKING:
    from collections.abc import Mapping

    from bilt.symbolic._dimensions import Dim
    from bilt.symbolic._tree import Tree


@dataclasses.dataclass(frozen=True)
class _Presence:
    """An axis is present if fixed or present in any of the masked unknowns."""

    mask: int = 0
    fixed: bool = False

    def combine(self, other: _Presence) -> _Presence:
        return _Presence(self.mask | other.mask, self.fixed or other.fixed)

    def value(self, included: int, excluded: int) -> bool | None:
        if self.fixed or self.mask & included:
            return True
        return None if self.mask & ~excluded else False


@dataclasses.dataclass(frozen=True)
class _Axis:
    unknowns: int
    rows: tuple[_Presence, ...]
    required: tuple[_Presence, ...]

    def propagate(self, included: int, excluded: int) -> tuple[int, int] | None:
        for required in self.required:
            state = required.value(included, excluded)
            if state is False:
                return None
            remaining = required.mask & ~excluded
            if state is None and remaining.bit_count() == 1:
                included |= remaining
        states = [row.value(included, excluded) for row in self.rows]
        row_min = sum(state is True for state in states)
        row_max = sum(state is not False for state in states)
        unknown_min = (self.unknowns & included).bit_count()
        unknown_max = (self.unknowns & ~excluded).bit_count()
        if row_min > unknown_max or unknown_min > row_max:
            return None
        if row_min == unknown_max:
            included |= self.unknowns & ~excluded
        if row_max == unknown_min:
            excluded |= self.unknowns & ~included
        return included, excluded


@dataclasses.dataclass
class _Constraints:
    names: tuple[str, ...]
    dims: tuple[Dim, ...]
    axes: tuple[_Axis, ...]

    @property
    def all_bits(self) -> int:
        return (1 << (len(self.names) * len(self.dims))) - 1

    def domains(self, included: int) -> dict[str, Domain]:
        width = len(self.names)
        return {
            name: Domain(tuple(dim for axis, dim in enumerate(self.dims) if included & (1 << (axis * width + index))))
            for index, name in enumerate(self.names)
        }

    def square(self, included: int) -> bool:
        """Compare complete row domains, not only scalar counts per axis."""
        width = len(self.names)
        outputs = [0] * width
        rows = [0] * width
        for axis, constraint in enumerate(self.axes):
            for index, presence in enumerate(constraint.rows):
                if presence.value(included, self.all_bits ^ included):
                    rows[index] |= 1 << axis
                if included & (1 << (axis * width + index)):
                    outputs[index] |= 1 << axis
        return collections.Counter(outputs) == collections.Counter(rows)

    def propagate(self, included: int, excluded: int, omit: int) -> tuple[int, int] | None:
        while True:
            previous = included, excluded
            if omit and not omit & excluded:
                remaining = omit & ~included
                if not remaining:
                    return None
                if remaining.bit_count() == 1:
                    excluded |= remaining
            for axis in self.axes:
                settled = axis.propagate(included, excluded)
                if settled is None:
                    return None
                included, excluded = settled
            if included & excluded:
                return None
            if previous == (included, excluded):
                return included, excluded

    def find(self, included: int, excluded: int, *, omit: int = 0) -> int | None:
        """Find a minimal layout; omit requires at least one named bit to be absent."""
        pending = [(included, excluded)]
        while pending:
            settled = self.propagate(*pending.pop(), omit)
            if settled is None:
                continue
            included, excluded = settled
            # Unassigned axes are absent in the candidate. This commonly proves
            # the least layout directly, without exploring other assignments.
            if self.valid(included, omit):
                return included
            remaining = self.all_bits & ~(included | excluded)
            if remaining:
                bit = remaining & -remaining
                pending.append((included | bit, excluded))
                pending.append((included, excluded | bit))
        return None

    def valid(self, included: int, omit: int) -> bool:
        excluded = self.all_bits ^ included
        return (
            (not omit or included & omit != omit)
            and all(required.value(included, excluded) for axis in self.axes for required in axis.required)
            and self.square(included)
        )


def infer_linear_domains(
    names: tuple[str, ...],
    rows: tuple[tuple[Tree, Tree], ...],
    known: Mapping[str, Domain],
    hints: Mapping[str, Domain],
    canonical: Domain,
) -> dict[str, Domain]:
    """Choose a layout contained in every other supported layout, or request hints.

    Only named axes present in known inputs are available. Domain hints are exact.
    Square layouts follow the joint assembler's equation/unknown domain multiset
    rule, independently of coordinate counts and matching row order.
    """
    constraints = _compile_constraints(names, rows, known, canonical)
    dims = constraints.dims
    width = len(names)
    included = excluded = 0
    for axis, dim in enumerate(dims):
        for index, name in enumerate(names):
            if name in hints:
                bit = 1 << (axis * width + index)
                if dim in hints[name].dims:
                    included |= bit
                else:
                    excluded |= bit
    settled = constraints.propagate(included, excluded, 0)
    solution = None if settled is None else constraints.find(*settled)
    if solution is None or settled is None:
        message = "No consistent linear-block domain layout satisfies the equations and Domain hints."
        raise BiltLinearBlockError(message)
    # A candidate composed entirely of forced bits is already the least layout.
    alternatives = solution & ~settled[0]
    if alternatives and constraints.find(included, excluded, omit=alternatives) is not None:
        ports = ", ".join(names)
        message = f"Ambiguous linear-block domains for {ports}. Add Domain hints to select a layout."
        raise BiltLinearBlockError(message)
    return constraints.domains(solution)


def _compile_constraints(
    names: tuple[str, ...], rows: tuple[tuple[Tree, Tree], ...], known: Mapping[str, Domain], canonical: Domain
) -> _Constraints:
    available = {dim for domain in known.values() for dim in domain.dims}
    dims = tuple(dim for dim in canonical.dims if dim in available)
    width = len(names)
    axes = []
    for axis, dim in enumerate(dims):
        variables = {name: 1 << (axis * width + index) for index, name in enumerate(names)}
        compiler = _AxisCompiler(dim, variables, known, available)
        presence = tuple(compiler.presence(left).combine(compiler.presence(right)) for left, right in rows)
        axes.append(_Axis(sum(variables.values()), presence, tuple(compiler.required)))
    if not dims:
        # Still inspect operators: an unbound reduction cannot invent an extent.
        compiler = _AxisCompiler(None, {}, known, available)
        for left, right in rows:
            compiler.presence(left)
            compiler.presence(right)
    return _Constraints(names, dims, tuple(axes))


@dataclasses.dataclass
class _AxisCompiler:
    dim: Dim | None
    variables: Mapping[str, int]
    known: Mapping[str, Domain]
    available: set[Dim]
    required: list[_Presence] = dataclasses.field(default_factory=list)
    cache: dict[Tree, _Presence] = dataclasses.field(default_factory=dict)

    def presence(self, tree: Tree) -> _Presence:
        if tree not in self.cache:
            self.cache[tree] = self._presence(tree)
        return self.cache[tree]

    def _presence(self, tree: Tree) -> _Presence:
        if tree.is_leaf:
            if tree.head.atom_type is SymbolType:
                name = typing.cast("str", tree.head.value)
                if name in self.variables:
                    return _Presence(self.variables[name])
                return _Presence(fixed=name in self.known and self.dim in self.known[name].dims)
            return _Presence()
        if tree.head.atom_type in (ReductionOpType, IndexOpType):
            axis = tree.args[1]
            if axis.head.atom_type is not DimensionType or axis.head.value not in self.available:
                message = (
                    f"Cannot infer linear-block domains: {tree.head} requires a dimension "
                    f"present in the block inputs (got {axis.head.value})."
                )
                raise BiltLinearBlockError(message)
            value = self.presence(tree.args[0])
            if axis.head.value == self.dim:
                self.required.append(value)
                return _Presence()
            return value
        result = _Presence()
        for child in tree.args:
            result = result.combine(self.presence(child))
        return result

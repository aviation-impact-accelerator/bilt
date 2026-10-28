"""Extract exact linear tensor operators without probing model code or numerical values."""

import dataclasses
import typing

from bilt.core._equation import BasicEquation
from bilt.core._structural_analysis._linear_block import BiltLinearBlockError, LinearBlock, extract_linear_block
from bilt.symbolic._algebra._affine import AffineAnalysisError
from bilt.symbolic._algebra._expr_ops import BINARY_ARG_COUNT, add, multiply, zero
from bilt.symbolic._atoms import ADD, ISEL, MEAN, MUL, NEG, SEL, SUM, CoordinateType, IntType, SymbolType
from bilt.symbolic._tree import Tree, symbol_tree

if typing.TYPE_CHECKING:
    from bilt.core._structural_analysis._plan import PlanBlock


@dataclasses.dataclass(frozen=True)
class LinearOperatorBlock:
    """Original linear equalities whose operators can couple tensor coordinates.

    Keep each complete receiver so broadcasting precedes reductions and selections.
    """

    block: PlanBlock
    unknowns: tuple[str, ...]
    equations: tuple[tuple[Tree, Tree], ...]

    @property
    def inputs(self) -> frozenset[str]:
        return self.block.inputs


def extract_supported_linear_block(block: PlanBlock) -> LinearBlock | LinearOperatorBlock:
    """Prefer pointwise coefficients; retain supported linear tensor operations."""
    try:
        return extract_linear_block(block)
    except BiltLinearBlockError as error:
        if not isinstance(error.__cause__, AffineAnalysisError):
            raise
    steps = tuple(sorted(block.steps, key=lambda step: step.output))
    unknowns = tuple(step.output for step in steps)
    equations = []
    seen = set()
    for step in steps:
        equation = step.operation
        if not isinstance(equation, BasicEquation) or id(equation) in seen:
            message = "Linear operator blocks require distinct primitive equations."
            raise BiltLinearBlockError(message)
        seen.add(id(equation))
        tree = equation.__bilt_equation_tree__
        if (
            not tree.is_equality
            or len(tree.args) != BINARY_ARG_COUNT
            or step.output not in equation.__bilt_parameters__
        ):
            message = "Linear operator blocks require binary equalities and declared outputs."
            raise BiltLinearBlockError(message)
        try:
            _validate_linear(tree.left, frozenset(unknowns))
            _validate_linear(tree.right, frozenset(unknowns))
        except AffineAnalysisError as error:
            message = f"Cannot extract supported linear tensor operators for {equation.__bilt_name__!r}."
            raise BiltLinearBlockError(message) from error
        equations.append((tree.left, tree.right))
    return LinearOperatorBlock(block, unknowns, tuple(equations))


def _depends(tree: Tree, symbols: frozenset[str]) -> bool:
    return (tree.is_leaf and tree.head.atom_type is SymbolType and tree.head.value in symbols) or any(
        _depends(child, symbols) for child in tree.args
    )


def _validate_linear(tree: Tree, symbols: frozenset[str]) -> None:
    if not _depends(tree, symbols) or tree.is_leaf:
        return
    if tree.head in (ADD, NEG):
        for child in tree.args:
            _validate_linear(child, symbols)
        return
    if tree.head is MUL:
        dependent = [child for child in tree.args if _depends(child, symbols)]
        if len(dependent) == 1:
            _validate_linear(dependent[0], symbols)
            return
    if tree.head in (SUM, MEAN, ISEL, SEL) and not any(_depends(child, symbols) for child in tree.args[1:]):
        _validate_selection(tree)
        _validate_linear(tree.args[0], symbols)
        return
    message = f"Unsupported unknown-dependent linear operator: {tree!s}."
    raise AffineAnalysisError(message)


def _validate_selection(tree: Tree) -> None:
    if tree.head not in (ISEL, SEL):
        return
    selector = tree.args[-1]
    selector_type = IntType if tree.head is ISEL else CoordinateType
    if not selector.is_leaf or selector.head.atom_type is not selector_type:
        message = "Linear unknown selections require a literal position or coordinate."
        raise AffineAnalysisError(message)


def linear_operator_rows(linear: LinearBlock) -> tuple[Tree, ...]:
    """Reconstruct pointwise linear actions for a joint layout with explicit hints."""
    rows = []
    for row in linear.coefficients:
        tree = zero()
        for coefficient, name in zip(row, linear.unknowns, strict=True):
            tree = add(tree, multiply(coefficient, symbol_tree(name)))
        rows.append(tree)
    return tuple(rows)

"""Describe equation residuals without choosing a numerical solver or backend."""

__all__ = ["BiltResidualBlockError", "ResidualBlock", "extract_residual_block"]

import dataclasses

from bilt.core._equation import BasicEquation
from bilt.core._errors import BiltAnalysisError
from bilt.core._structural_analysis._plan import PlanBlock
from bilt.core._structural_analysis._qualified_block import QualifiedBlock, QualifiedEquation
from bilt.symbolic._algebra._expr_ops import BINARY_ARG_COUNT, UNARY_ARG_COUNT
from bilt.symbolic._atoms import ADD, INV, MUL, NEG, POW, FloatType, IntType, QuantityType, SymbolType
from bilt.symbolic._tree import Tree


class BiltResidualBlockError(BiltAnalysisError):
    """An equation block cannot be represented by supported residual expressions."""


@dataclasses.dataclass(frozen=True)
class ResidualBlock:
    """Ordered ``lhs - rhs`` expressions retaining their original equation rows.

    Local rows and unknowns follow sorted matched output order, as in linear
    extraction. Qualified blocks retain the caller's independent row and column
    orders, including rectangular blocks. ``rows`` retain source identity, path
    and display label. Inputs are declared dependencies outside the unknowns.

    This is a symbolic description only: it performs no isolation, domain/shape
    analysis, differentiation, rank checks or numerical execution.
    """

    block: PlanBlock | QualifiedBlock
    unknowns: tuple[str, ...]
    rows: tuple[QualifiedEquation, ...]
    residuals: tuple[Tree, ...]

    @property
    def inputs(self) -> frozenset[str]:
        return self.block.inputs


def extract_residual_block(block: PlanBlock | QualifiedBlock) -> ResidualBlock:
    """Construct residuals from primitive equalities, independently of linear extraction.

    Supported expressions contain numeric/quantity constants, variable symbols,
    addition, multiplication, negation, reciprocal and power. Reductions,
    selections, nested relations and other operations are rejected everywhere,
    including expressions depending only on external inputs. Transforms and
    opaque subsystem routes must not be passed as equation rows.

    No simplification is performed: even identical sides are retained so that
    construction cannot hide unsupported operations or erase dependencies.
    """
    if isinstance(block, PlanBlock):
        steps = tuple(sorted(block.steps, key=lambda step: step.output))
        unknowns = tuple(step.output for step in steps)
        if len(set(unknowns)) != len(unknowns):
            message = "Cannot construct residuals with duplicate output assignments."
            raise BiltResidualBlockError(message)
        local_rows = []
        seen: set[int] = set()
        for step in steps:
            source = step.operation
            if not isinstance(source, BasicEquation):
                message = f"Cannot construct residual for {step.output!r}: {type(source).__name__} is not a primitive equation."
                raise BiltResidualBlockError(message)
            if id(source) in seen:
                message = f"Cannot use equation {source.__bilt_name__!r} more than once in a residual block."
                raise BiltResidualBlockError(message)
            seen.add(id(source))
            if step.output not in source.__bilt_parameters__:
                message = f"Equation {source.__bilt_name__!r} does not declare block output {step.output!r}."
                raise BiltResidualBlockError(message)
            local_rows.append(QualifiedEquation(source))
        rows = tuple(local_rows)
    else:
        unknowns = block.unknowns
        rows = block.rows

    residuals = []
    for row in rows:
        tree = row.tree
        if not tree.is_equality or len(tree.args) != BINARY_ARG_COUNT:
            message = f"Equation {row.label!r} is not a binary equality."
            raise BiltResidualBlockError(message)
        _validate_expression(tree.left, row)
        _validate_expression(tree.right, row)
        residuals.append(Tree(ADD, (tree.left, Tree(NEG, (tree.right,)))))
    return ResidualBlock(block, unknowns, rows, tuple(residuals))


def _validate_expression(tree: Tree, row: QualifiedEquation) -> None:
    if tree.is_leaf and tree.head.atom_type in (SymbolType, IntType, FloatType, QuantityType):
        if tree.head.atom_type is SymbolType and tree.head.value not in row.parameters:
            message = f"Equation {row.label!r} contains undeclared symbol {tree.head.value!r}."
            raise BiltResidualBlockError(message)
        return
    arity = len(tree.args)
    supported = (
        (tree.head in (ADD, MUL) and arity >= BINARY_ARG_COUNT)
        or (tree.head in (NEG, INV) and arity == UNARY_ARG_COUNT)
        or (tree.head is POW and arity == BINARY_ARG_COUNT)
    )
    if not supported:
        message = (
            f"Equation {row.label!r} contains unsupported residual operation {tree.head!r} with {arity} arguments."
        )
        raise BiltResidualBlockError(message)
    for child in tree.args:
        _validate_expression(child, row)

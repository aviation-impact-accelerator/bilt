"""Extract exact linear tensor operators without probing model code or numerical values."""

import dataclasses
import typing

from bilt.core._equation import BasicEquation
from bilt.core._structural_analysis._linear_block import BiltLinearBlockError, LinearBlock, extract_linear_block
from bilt.symbolic._algebra._affine import AffineAnalysisError, AffineForm, add_affine, scale_affine, subtract_affine
from bilt.symbolic._algebra._expr_ops import BINARY_ARG_COUNT, is_zero, multiply_many, negate, one, zero
from bilt.symbolic._atoms import ADD, MEAN, MUL, NEG, SUM, SymbolType
from bilt.symbolic._tree import Tree

if typing.TYPE_CHECKING:
    from bilt.core._structural_analysis._plan import PlanBlock


@dataclasses.dataclass(frozen=True)
class LinearOperatorBlock:
    """Rows of linear actions on unknown tensors, with independent right-hand sides.

    Unlike pointwise coefficients, each operator tree includes the unknown on
    which it acts. Reductions can couple different coordinates.
    """

    block: PlanBlock
    unknowns: tuple[str, ...]
    operators: tuple[tuple[Tree, ...], ...]
    rhs: tuple[Tree, ...]

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
    operators = []
    rhs = []
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
            form = subtract_affine(
                _operator_form(tree.left, frozenset(unknowns)), _operator_form(tree.right, frozenset(unknowns))
            )
        except AffineAnalysisError as error:
            message = f"Cannot extract supported linear tensor operators for {equation.__bilt_name__!r}."
            raise BiltLinearBlockError(message) from error
        operators.append(tuple(form.coefficients.get(name, zero()) for name in unknowns))
        rhs.append(negate(form.constant))
    return LinearOperatorBlock(block, unknowns, tuple(operators), tuple(rhs))


def _depends(tree: Tree, symbols: frozenset[str]) -> bool:
    return (tree.is_leaf and tree.head.atom_type is SymbolType and tree.head.value in symbols) or any(
        _depends(child, symbols) for child in tree.args
    )


def _operator_form(tree: Tree, symbols: frozenset[str]) -> AffineForm:
    if not _depends(tree, symbols):
        return AffineForm({}, tree)
    if tree.is_leaf:
        return AffineForm({str(tree.head.value): tree}, zero())
    if tree.head is ADD:
        result = AffineForm({}, zero())
        for child in tree.args:
            result = add_affine(result, _operator_form(child, symbols))
        return result
    if tree.head is NEG:
        return scale_affine(_operator_form(tree.args[0], symbols), negate(one()))
    if tree.head is MUL:
        dependent = [child for child in tree.args if _depends(child, symbols)]
        if len(dependent) == 1:
            return scale_affine(
                _operator_form(dependent[0], symbols),
                multiply_many(child for child in tree.args if child is not dependent[0]),
            )
    if tree.head in (SUM, MEAN) and not any(_depends(child, symbols) for child in tree.args[1:]):
        form = _operator_form(tree.args[0], symbols)
        return AffineForm(
            {name: Tree(tree.head, (value, *tree.args[1:])) for name, value in form.coefficients.items()},
            zero() if is_zero(form.constant) else Tree(tree.head, (form.constant, *tree.args[1:])),
        )
    message = f"Unsupported unknown-dependent linear operator: {tree!s}."
    raise AffineAnalysisError(message)

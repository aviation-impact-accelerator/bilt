"""Extract symbolic linear systems from blocks of primitive equations."""

__all__ = ["BiltLinearBlockError", "LinearBlock", "extract_linear_block", "extract_qualified_linear_block"]

import dataclasses
import typing

from bilt.core._equation import BasicEquation
from bilt.core._errors import BiltAnalysisError
from bilt.core._structural_analysis._plan import PlanBlock
from bilt.symbolic._algebra._affine import AffineAnalysisError, as_affine, subtract_affine
from bilt.symbolic._algebra._expr_ops import BINARY_ARG_COUNT, negate, zero

if typing.TYPE_CHECKING:
    from bilt.core._structural_analysis._plan import PlanStep
    from bilt.core._structural_analysis._qualified_block import QualifiedBlock
    from bilt.symbolic._tree import Tree


class BiltLinearBlockError(BiltAnalysisError):
    """A block has no supported affine extraction.

    This does not prove mathematical nonlinearity or rule out other execution
    strategies. The original affine-analysis error is retained as the cause.
    """


@dataclasses.dataclass(frozen=True)
class LinearBlock:
    """Symbolic ``coefficients @ unknowns = rhs``, retaining the source block.

    Columns follow ``unknowns`` in sorted output order. Rows follow
    ``row_steps`` in the same order, independently of the source step order.
    Coefficients and right-hand sides contain no block unknowns, but may depend
    nonlinearly on external inputs. No rank, numerical or tensor-shape analysis
    is performed here.
    """

    block: PlanBlock | QualifiedBlock
    unknowns: tuple[str, ...]
    coefficients: tuple[tuple[Tree, ...], ...]
    rhs: tuple[Tree, ...]

    @property
    def row_steps(self) -> tuple[PlanStep, ...]:
        """Return the original matched steps corresponding to the matrix rows."""
        if not isinstance(self.block, PlanBlock):
            message = "Qualified linear rows retain their sources in block.rows, rather than local plan steps."
            raise TypeError(message)
        return tuple(sorted(self.block.steps, key=lambda step: step.output))

    @property
    def inputs(self) -> frozenset[str]:
        """Return external dependencies retained from the source block."""
        return self.block.inputs


def extract_linear_block(block: PlanBlock) -> LinearBlock:
    """Extract coefficients from original equalities without isolating them.

    Only primitive equation steps are supported. Transforms and opaque subsystem
    routes are rejected; child equations must be expanded and qualified by the
    caller before they can participate in a joint linear system.

    Unsupported affine expressions raise ``BiltLinearBlockError``. Zero rows
    and linearly dependent equations are retained for later solve diagnostics.
    """
    steps = tuple(sorted(block.steps, key=lambda step: step.output))
    unknowns = tuple(step.output for step in steps)
    symbols = frozenset(unknowns)
    if len(symbols) != len(steps):
        message = "Cannot extract a linear block with duplicate output assignments."
        raise BiltLinearBlockError(message)

    coefficients: list[tuple[Tree, ...]] = []
    rhs: list[Tree] = []
    seen: set[int] = set()

    for step in steps:
        equation = step.operation
        if not isinstance(equation, BasicEquation):
            message = (
                f"Cannot extract a linear block for {step.output!r}: "
                f"{type(equation).__name__} is not a primitive equation."
            )
            raise BiltLinearBlockError(message)
        if id(equation) in seen:
            message = f"Cannot use equation {equation.__bilt_name__!r} more than once in a linear block."
            raise BiltLinearBlockError(message)
        seen.add(id(equation))
        if step.output not in equation.__bilt_parameters__:
            message = f"Equation {equation.__bilt_name__!r} does not declare block output {step.output!r}."
            raise BiltLinearBlockError(message)

        tree = equation.__bilt_equation_tree__
        if not tree.is_equality or len(tree.args) != BINARY_ARG_COUNT:
            message = f"Equation {equation.__bilt_name__!r} is not a binary equality."
            raise BiltLinearBlockError(message)
        try:
            form = subtract_affine(as_affine(tree.left, symbols), as_affine(tree.right, symbols))
        except AffineAnalysisError as error:
            message = (
                f"Cannot extract a supported affine form for equation {equation.__bilt_name__!r} "
                f"with block unknowns {unknowns!r}."
            )
            raise BiltLinearBlockError(message) from error

        coefficients.append(tuple(form.coefficients.get(symbol, zero()) for symbol in unknowns))
        rhs.append(negate(form.constant))

    return LinearBlock(
        block=block,
        unknowns=unknowns,
        coefficients=tuple(coefficients),
        rhs=tuple(rhs),
    )


def extract_qualified_linear_block(block: QualifiedBlock) -> LinearBlock:
    """Extract a square scalar affine system from source-preserving qualified rows."""
    if len(block.rows) != len(block.unknowns):
        message = "A qualified linear solve requires equally many primitive rows and unknowns."
        raise BiltLinearBlockError(message)
    coefficients = []
    rhs = []
    symbols = frozenset(block.unknowns)
    for row in block.rows:
        tree = row.tree
        if not tree.is_equality or len(tree.args) != BINARY_ARG_COUNT:
            message = f"Equation {row.label!r} is not a binary equality."
            raise BiltLinearBlockError(message)
        try:
            form = subtract_affine(as_affine(tree.left, symbols), as_affine(tree.right, symbols))
        except AffineAnalysisError as error:
            message = (
                f"Cannot extract a supported affine form for equation {row.label!r} with unknowns {block.unknowns!r}."
            )
            raise BiltLinearBlockError(message) from error
        coefficients.append(tuple(form.coefficients.get(name, zero()) for name in block.unknowns))
        rhs.append(negate(form.constant))
    return LinearBlock(block, block.unknowns, tuple(coefficients), tuple(rhs))

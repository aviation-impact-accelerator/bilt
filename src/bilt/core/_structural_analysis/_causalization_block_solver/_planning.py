"""Construct an ordered execution plan from candidate assignments."""

__all__ = ["plan_blocks"]

import typing

from bilt.core._errors import BiltAnalysisError
from bilt.core._evaluation_options import current_root_options
from bilt.core._structural_analysis._causalization_block_solver._blocks import ordered_blocks
from bilt.core._structural_analysis._causalization_block_solver._dependency_graph import build_dependency_graph
from bilt.core._structural_analysis._causalization_block_solver._errors import BiltExecutionPlanError
from bilt.core._structural_analysis._causalization_block_solver._settle_assignments import settle_assignments
from bilt.core._structural_analysis._linear_block import BiltLinearBlockError
from bilt.core._structural_analysis._linear_operator import extract_supported_linear_block
from bilt.core._structural_analysis._realize_nonlinear import nonlinear_candidate

if typing.TYPE_CHECKING:
    import collections.abc

    from bilt._backend import Backend
    from bilt.core._model_constituent import BiltModelConstituent
    from bilt.core._structural_analysis._plan import PlanBlock, PlanStep


def plan_blocks(
    assignments: collections.abc.Mapping[str, BiltModelConstituent],
    available: set[str],
    output: str,
    *,
    backend: Backend,
) -> tuple[PlanBlock, ...]:
    """Select target dependencies and preserve supported ordered blocks."""
    if output in available:
        return ()

    if output not in assignments:
        msg = f"No candidate producer for {output!r}."
        raise BiltExecutionPlanError(msg)

    settled, _ = settle_assignments(
        assignments,
        available,
        backend,
        target=output,
    )
    by_output = {step.output: step for step in settled}
    selected: dict[str, PlanStep] = {}
    pending = [output]

    while pending:
        attribute = pending.pop()

        if attribute in available or attribute in selected:
            continue

        step = by_output.get(attribute)
        if step is None:
            msg = f"No supported execution route for {attribute!r}."
            raise BiltExecutionPlanError(msg)

        selected[attribute] = step
        pending.extend(step.consumed)

    graph, unresolved = build_dependency_graph(
        tuple(selected.values()),
        available,
    )
    if unresolved:
        msg = f"Unresolved execution dependencies: {unresolved}."
        raise BiltExecutionPlanError(msg)

    blocks = ordered_blocks(graph)
    for block in blocks:
        if block.is_cyclic:
            _check_block(block)

    return blocks


def _check_block(block: PlanBlock) -> None:
    """Try independent strategies before declaring a formulation unsupported."""
    try:
        try:
            extract_supported_linear_block(block)
        except BiltLinearBlockError:
            if current_root_options() is None:
                raise
            nonlinear_candidate(block)
    except BiltAnalysisError as error:
        outputs = ", ".join(sorted(block.outputs))
        msg = (
            f"Execution of coupled blocks is not yet supported for this formulation. "
            f"Required computations: {outputs}. {error}"
        )
        raise BiltExecutionPlanError(msg) from error

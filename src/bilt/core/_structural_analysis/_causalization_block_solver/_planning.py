"""Construct an ordered execution plan from candidate assignments."""

__all__ = ["plan_blocks"]

import typing

from bilt.core._structural_analysis._causalization_block_solver._blocks import ordered_blocks
from bilt.core._structural_analysis._causalization_block_solver._dependency_graph import build_dependency_graph
from bilt.core._structural_analysis._causalization_block_solver._errors import BiltExecutionPlanError
from bilt.core._structural_analysis._causalization_block_solver._settle_assignments import settle_assignments

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
    """Select target dependencies and preserve their ordered acyclic blocks."""
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
    if any(block.is_cyclic for block in blocks):
        outputs = ", ".join(sorted(selected))
        msg = f"Execution of coupled blocks is not yet supported by this analyser. Required computations: {outputs}."
        raise BiltExecutionPlanError(msg)

    return blocks

"""Construct an ordered execution plan from candidate assignments."""

__all__ = ["plan_blocks", "plan_blocks_many"]

import typing

from bilt.core._structural_analysis._causalization_block_solver._blocks import ordered_blocks
from bilt.core._structural_analysis._causalization_block_solver._dependency_graph import build_dependency_graph
from bilt.core._structural_analysis._causalization_block_solver._errors import BiltExecutionPlanError
from bilt.core._structural_analysis._causalization_block_solver._settle_assignments import settle_assignments_many
from bilt.core._structural_analysis._linear_block import BiltLinearBlockError
from bilt.core._structural_analysis._linear_operator import extract_supported_linear_block

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
    """Retain the original single-output route behavior."""
    return plan_blocks_many(assignments, available, (output,), backend=backend)


def plan_blocks_many(
    assignments: collections.abc.Mapping[str, BiltModelConstituent],
    available: set[str],
    outputs: tuple[str, ...],
    *,
    backend: Backend,
) -> tuple[PlanBlock, ...]:
    """Plan all targets using one matching, with shared dependency blocks."""
    if not outputs:
        msg = "Multi-output planning needs at least one output."
        raise ValueError(msg)
    needed = tuple(output for output in outputs if output not in available)
    if not needed:
        return ()
    for output in needed:
        if output not in assignments:
            msg = f"No candidate producer for {output!r}."
            raise BiltExecutionPlanError(msg)

    settled, _ = settle_assignments_many(
        assignments, available, backend, targets=needed
    )
    by_output = {step.output: step for step in settled}
    selected: dict[str, PlanStep] = {}
    pending = list(needed)

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

    graph, unresolved = build_dependency_graph(tuple(selected.values()), available)
    if unresolved:
        msg = f"Unresolved execution dependencies: {unresolved}."
        raise BiltExecutionPlanError(msg)

    blocks = ordered_blocks(graph)
    for block in blocks:
        if not block.is_cyclic:
            continue
        try:
            extract_supported_linear_block(block)
        except BiltLinearBlockError as error:
            names = ", ".join(sorted(block.outputs))
            msg = (
                "Execution of coupled blocks is not yet supported for this formulation. "
                f"Required computations: {names}. {error}"
            )
            raise BiltExecutionPlanError(msg) from error
    return blocks

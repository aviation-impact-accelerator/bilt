"""Construct an ordered execution plan from candidate assignments."""

__all__ = ["plan_steps"]

import typing

import rustworkx as rx

from bilt.core._structural_analysis._causalization_block_solver._blocks import ordered_blocks
from bilt.core._structural_analysis._causalization_block_solver._dependency_graph import build_dependency_graph
from bilt.core._structural_analysis._causalization_block_solver._errors import BiltExecutionPlanError
from bilt.core._structural_analysis._causalization_block_solver._settle_assignments import settle_assignments

if typing.TYPE_CHECKING:
    import collections.abc

    from bilt._backend import Backend
    from bilt.core._model_constituent import BiltModelConstituent
    from bilt.core._structural_analysis._plan import PlanStep


def plan_steps(
    assignments: collections.abc.Mapping[str, BiltModelConstituent],
    available: set[str],
    output: str,
    *,
    backend: Backend,
) -> tuple[PlanStep, ...]:
    """Select the target's dependencies and order an acyclic plan."""
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
    if not rx.is_directed_acyclic_graph(graph):
        outputs = ", ".join(sorted(selected))
        msg = f"Execution of coupled blocks is not yet supported by this analyser. Required computations: {outputs}."
        raise BiltExecutionPlanError(msg)

    return tuple(step for block in blocks for step in block)

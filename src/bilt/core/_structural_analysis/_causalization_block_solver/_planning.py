"""Construct an ordered execution plan from candidate assignments."""

__all__ = ["plan_blocks"]

import typing

from bilt.core._equation import BasicEquation
from bilt.core._structural_analysis._causalization_block_solver._blocks import ordered_blocks
from bilt.core._structural_analysis._causalization_block_solver._dependency_graph import build_dependency_graph
from bilt.core._structural_analysis._causalization_block_solver._errors import BiltExecutionPlanError
from bilt.core._structural_analysis._causalization_block_solver._settle_assignments import settle_assignments
from bilt.core._structural_analysis._linear_block import BiltLinearBlockError
from bilt.core._structural_analysis._linear_domains import equation_domain_hints
from bilt.core._structural_analysis._linear_operator import LinearOperatorBlock, extract_supported_linear_block
from bilt.core._structural_analysis._plan import PlanBlock, PlanStep
from bilt.symbolic._expressions import Expr
from bilt.symbolic._isolation._errors import BiltUnsupportedIsolationError
from bilt.symbolic._isolation._isolate import isolate

if typing.TYPE_CHECKING:
    import collections.abc

    from bilt._backend import Backend
    from bilt.core._model_constituent import BiltModelConstituent


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

    _include_related_linear_constraints(selected, assignments, available)
    _include_operator_boundary_constraints(selected, assignments, available)

    graph, unresolved = build_dependency_graph(
        tuple(selected.values()),
        available,
    )
    if unresolved:
        msg = f"Unresolved execution dependencies: {unresolved}."
        raise BiltExecutionPlanError(msg)

    blocks = _promote_linear_operator_chains(ordered_blocks(graph))
    for block in blocks:
        if not block.is_cyclic:
            continue
        try:
            extract_supported_linear_block(block)
        except BiltLinearBlockError as error:
            outputs = ", ".join(sorted(block.outputs))
            msg = (
                f"Execution of coupled blocks is not yet supported for this formulation. "
                f"Required computations: {outputs}. {error}"
            )
            raise BiltExecutionPlanError(msg) from error

    return blocks


def _requires_linear_operator(block: PlanBlock) -> bool:
    """A primitive equality is linear but cannot be inverted as a single expression."""
    if block.is_cyclic or len(block.steps) != 1:
        return False
    step = block.steps[0]
    if not isinstance(step.operation, BasicEquation):
        return False
    try:
        isolate(step.operation.__bilt_equation_tree__, Expr.symbol(step.output))
    except BiltUnsupportedIsolationError:
        return True
    return False


def _promote_linear_operator_chains(blocks: tuple[PlanBlock, ...]) -> tuple[PlanBlock, ...]:  # noqa: C901
    """Prototype: lower non-invertible linear equations, including short chains, as blocks.

    A reduction can be linear without being individually invertible. If its
    intermediate output is needed by the next primitive equation, retaining
    those equations as one block permits normal operator lowering to assemble
    and eliminate the intermediate numerically.

    This intentionally handles only one equation or a contiguous chain;
    branch/fan-out analysis is out of scope for this prototype.
    """
    pending = list(blocks)
    result: list[PlanBlock] = []
    while pending:
        block = pending.pop(0)
        if block.is_cyclic:
            try:
                linear = extract_supported_linear_block(block)
            except BiltLinearBlockError:
                linear = None
            if isinstance(linear, LinearOperatorBlock):
                chain = [block]
                outputs = set(block.outputs)
                while pending:
                    next_block = pending[0]
                    if (
                        next_block.is_cyclic
                        or len(next_block.steps) != 1
                        or not isinstance(next_block.steps[0].operation, BasicEquation)
                        or not (next_block.inputs & outputs)
                    ):
                        break
                    candidate = PlanBlock(
                        steps=tuple(step for part in (*chain, next_block) for step in part.steps),
                        is_cyclic=True,
                    )
                    try:
                        extract_supported_linear_block(candidate)
                    except BiltLinearBlockError:
                        break
                    pending.pop(0)
                    chain.append(next_block)
                    outputs.update(next_block.outputs)
                if len(chain) > 1:
                    result.append(
                        PlanBlock(
                            steps=tuple(step for part in chain for step in part.steps),
                            is_cyclic=True,
                        )
                    )
                    continue
        if not _requires_linear_operator(block):
            result.append(block)
            continue

        chain = [block]
        while (
            pending
            and not pending[0].is_cyclic
            and len(pending[0].steps) == 1
            and isinstance(pending[0].steps[0].operation, BasicEquation)
            and chain[-1].outputs & pending[0].inputs
        ):
            chain.append(pending.pop(0))

        candidate = PlanBlock(
            steps=tuple(step for part in chain for step in part.steps),
            is_cyclic=True,
        )
        try:
            extract_supported_linear_block(candidate)
        except BiltLinearBlockError:
            result.extend(chain)
        else:
            result.append(candidate)
    return tuple(result)


def _include_related_linear_constraints(
    selected: dict[str, PlanStep],
    assignments: collections.abc.Mapping[str, BiltModelConstituent],
    available: set[str],
) -> None:
    """Prototype: retain the neighbouring row that closes an inverse tensor operator.

    Matching is at the port level, rather than per tensor coordinate. A
    non-invertible reduction may be selected as the only producer for a
    tensor output even though another matched equation is required to
    constrain its individual coordinates. Extend such a one-row request
    with one directly connected matched equation, when the pair is linear.
    """
    if len(selected) != 1:
        return
    original = next(iter(selected.values()))
    if not _requires_linear_operator(PlanBlock((original,))):
        return
    for name, operation in sorted(assignments.items()):
        if name in selected or name in available or not isinstance(operation, BasicEquation):
            continue
        candidate = PlanStep(operation=operation, output=name)
        if original.output not in candidate.consumed:
            continue
        if any(dep not in available and dep != original.output for dep in candidate.consumed):
            continue
        block = PlanBlock((original, candidate), is_cyclic=True)
        try:
            extract_supported_linear_block(block)
        except BiltLinearBlockError:
            continue
        selected[name] = candidate
        return


def _include_operator_boundary_constraints(
    selected: dict[str, PlanStep],
    assignments: collections.abc.Mapping[str, BiltModelConstituent],
    available: set[str],
) -> None:
    """Prototype: close a cyclic tensor block with connected boundary operators.

    A scalar lift and moment balance can match wing and tail tensor ports,
    leaving the target-panel boundary equations as acyclic downstream rows.
    Those target rows are still necessary to determine each tensor coordinate.
    """
    if len(selected) <= 1:
        return
    graph, unresolved = build_dependency_graph(tuple(selected.values()), available)
    if unresolved:
        return
    if not any(block.is_cyclic for block in ordered_blocks(graph)):
        return
    for name, operation in sorted(assignments.items()):
        if name in selected or name in available or not isinstance(operation, BasicEquation):
            continue
        step = PlanStep(operation=operation, output=name)
        declarations = operation.__bilt_port_declarations__
        has_tensor_operator = any(
            declaration.required_domain is not None and len(declaration.required_domain.dims) > 1
            for declaration in declarations.values()
        ) or any(len(domain.dims) > 1 for _, domain in equation_domain_hints(operation))
        if not has_tensor_operator:
            continue
        if not (step.consumed & selected.keys()):
            continue
        if any(dep not in available and dep not in selected for dep in step.consumed):
            continue
        selected[step.output] = step

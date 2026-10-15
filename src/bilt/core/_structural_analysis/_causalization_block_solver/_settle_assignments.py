__all__ = ["settle_assignments"]

import typing

from bilt.core._structural_analysis._plan import PlanStep
from bilt.core._sub_system import SubSystemConstituent

if typing.TYPE_CHECKING:
    import collections.abc

    from bilt._backend._base_backend import Backend
    from bilt.core._model_constituent import BiltModelConstituent


def settle_assignments(
    assignments: collections.abc.Mapping[str, BiltModelConstituent],
    available: collections.abc.Set[str],
    backend: Backend,
) -> tuple[tuple[PlanStep, ...], frozenset[str]]:
    """Propagate established dependencies in stable order until no progress.

    After stalling, retain remaining candidate dependencies for explicit cycle
    and unresolved-input checks. A rejected child route is local to this
    candidate input set, not a permanent rejection of the child output.
    """
    established = set(available)
    pending = dict(assignments)
    settled: dict[str, PlanStep] = {}

    while pending:
        before = len(pending)
        for output in sorted(pending):
            step = _assignment_step(pending[output], output, established, backend)
            if step is not None and set(step.consumed) <= established:
                settled[output] = step
                established.add(output)
        for output in settled:
            pending.pop(output, None)
        if len(pending) == before:
            break

    candidate_inputs = set(available) | set(assignments)
    for output, operation in sorted(pending.items()):
        step = _assignment_step(operation, output, candidate_inputs, backend)
        if step is not None:
            settled[output] = step

    return tuple(settled[output] for output in sorted(settled)), frozenset(assignments.keys() - settled.keys())


def _assignment_step(
    operation: BiltModelConstituent,
    output: str,
    available: collections.abc.Set[str],
    backend: Backend,
) -> PlanStep | None:
    if not isinstance(operation, SubSystemConstituent):
        return PlanStep(operation=operation, output=output)

    candidate_ports = (set(operation.__bilt_parameters__) & available) - {output}
    child_inputs = operation.required_inputs_for(output, backend, within=candidate_ports)
    if child_inputs is None:
        return None
    return PlanStep(operation=operation, output=output, child_inputs=child_inputs)

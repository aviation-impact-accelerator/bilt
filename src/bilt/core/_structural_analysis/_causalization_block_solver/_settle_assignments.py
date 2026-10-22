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
    *,
    target: str,
) -> tuple[tuple[PlanStep, ...], frozenset[str]]:
    """Settle the target's dependencies until no more inputs are established.

    After stalling, retain remaining candidate dependencies for explicit cycle
    and unresolved-input checks. A rejected child route is local to this
    candidate input set, not a permanent rejection of the child output.
    Follow the target's dependencies and potential inputs of selected children.
    """
    return _TargetSettlement(assignments, available, backend).settle(target)


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


class _TargetSettlement:
    """Request-local dependency traversal; unestablished candidates retain cycle evidence."""

    def __init__(
        self,
        assignments: collections.abc.Mapping[str, BiltModelConstituent],
        available: collections.abc.Set[str],
        backend: Backend,
    ) -> None:
        self.assignments = assignments
        self.backend = backend
        self.established = set(available)
        self.candidates = self.established | set(assignments)
        self.steps: dict[str, PlanStep] = {}
        self.requested: set[str] = set()
        self.active: set[str] = set()

    def settle(self, output: str) -> tuple[tuple[PlanStep, ...], frozenset[str]]:
        """Revisit stalled dependencies whenever another value was established."""
        while output not in self.established:
            before = len(self.established)
            pending = [self._resolve(output)]
            while pending:
                try:
                    dependency = next(pending[-1])
                except StopIteration:
                    pending.pop()
                else:
                    pending.append(self._resolve(dependency))
            if len(self.established) == before:
                break
        return (
            tuple(self.steps[name] for name in sorted(self.steps)),
            frozenset((self.requested & self.assignments.keys()) - self.steps.keys()),
        )

    def _resolve(self, attribute: str) -> collections.abc.Iterator[str]:
        if attribute in self.established or attribute in self.active:
            return
        self.requested.add(attribute)
        operation = self.assignments.get(attribute)
        if operation is None:
            return
        self.active.add(attribute)
        try:
            if isinstance(operation, SubSystemConstituent):
                step = yield from self._child_step(operation, attribute)
            else:
                step = PlanStep(operation=operation, output=attribute)
            if step is None:
                self.steps.pop(attribute, None)
                return
            self.steps[attribute] = step
            yield from step.consumed
            if set(step.consumed) <= self.established:
                self.established.add(attribute)
        finally:
            self.active.remove(attribute)

    def _child_step(
        self, operation: SubSystemConstituent, output: str
    ) -> collections.abc.Generator[str, None, PlanStep | None]:
        ports = sorted((set(operation.__bilt_parameters__) & self.candidates) - {output})
        # Establish parent-produced boundary inputs together before asking the
        # child to search with an unnecessarily incomplete set of supplied ports.
        for port in ports:
            if not isinstance(self.assignments.get(port), SubSystemConstituent):
                yield port
        step = _assignment_step(operation, output, self.established, self.backend)
        if step is not None:
            return step
        for port in ports:
            if not isinstance(self.assignments.get(port), SubSystemConstituent):
                continue
            before = len(self.established)
            yield port
            if len(self.established) != before:
                step = _assignment_step(operation, output, self.established, self.backend)
                if step is not None:
                    return step
        # Preserve cyclic/unresolved candidates for explicit checks, without
        # claiming their dependencies have been established.
        return _assignment_step(operation, output, self.candidates, self.backend)

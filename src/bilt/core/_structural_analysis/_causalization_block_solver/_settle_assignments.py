__all__ = ["settle_assignments"]

import typing

from bilt.core._structural_analysis._causalization_block_solver._errors import BiltOverconstrainedError
from bilt.core._structural_analysis._causalization_block_solver._structural_validation import (
    validate_request_structure,
)
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
        self.fixed = frozenset(available)
        self.established = set(available)
        self.candidates = self.established | set(assignments)
        self.steps: dict[str, PlanStep] = {}
        self.requested: set[str] = set()
        self.active: set[str] = set()

    def settle(self, output: str) -> tuple[tuple[PlanStep, ...], frozenset[str]]:
        """Revisit stalled dependencies whenever another value was established."""
        while output not in self.established:
            before = len(self.established)
            self._walk_target(output)
            if len(self.established) == before:
                break
        return (
            tuple(self.steps[name] for name in sorted(self.steps)),
            frozenset((self.requested & self.assignments.keys()) - self.steps.keys()),
        )

    def _walk_target(self, output: str) -> None:
        """Resolve dependencies using explicit frames instead of nested generators."""
        pending: list[tuple[str, collections.abc.Iterator[str], PlanStep | None]] = []

        def enter(attribute: str) -> None:
            if attribute in self.established or attribute in self.active:
                return
            self.requested.add(attribute)
            operation = self.assignments.get(attribute)
            if operation is None:
                return
            self.active.add(attribute)
            if isinstance(operation, SubSystemConstituent):
                pending.append((attribute, self._child_step(operation, attribute), None))
            else:
                step = PlanStep(operation=operation, output=attribute)
                self.steps[attribute] = step
                pending.append((attribute, iter(step.consumed), step))

        enter(output)
        try:
            while pending:
                attribute, dependencies, step = pending[-1]
                try:
                    dependency = next(dependencies)
                except StopIteration as finished:
                    if step is None:
                        selected: PlanStep | None = finished.value
                        if selected is not None:
                            self.steps[attribute] = selected
                            pending[-1] = (attribute, iter(selected.consumed), selected)
                            continue
                        self.steps.pop(attribute, None)
                    elif set(step.consumed) <= self.established:
                        self.established.add(attribute)
                    pending.pop()
                    self.active.remove(attribute)
                else:
                    enter(dependency)
        finally:
            # A child-route exception must not leave an attribute marked active.
            self.active.difference_update(attribute for attribute, _, _ in pending)

    def _child_step(
        self, operation: SubSystemConstituent, output: str
    ) -> collections.abc.Generator[str, None, PlanStep | None]:
        ports = sorted((set(operation.__bilt_parameters__) & self.candidates) - {output})
        # Establish parent-produced boundary inputs together before asking the
        # child to search with an unnecessarily incomplete set of supplied ports.
        for port in ports:
            if not isinstance(self.assignments.get(port), SubSystemConstituent):
                yield port
        step = self._child_route(operation, output)
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

    def _child_route(self, operation: SubSystemConstituent, output: str) -> PlanStep | None:
        """Avoid constraining a child with its own already-computed sibling outputs.

        Prefer the ordinary route. Only if the child's requested route is
        structurally overconstrained do we retry without outputs produced by
        this same child in the parent. Explicitly supplied ports stay fixed.
        """
        step = _assignment_step(operation, output, self.established, self.backend)
        if step is None or step.child_inputs is None:
            return step

        sibling_outputs = {
            port
            for port in self.established
            if port not in self.fixed and port != output and self.assignments.get(port) is operation
        }
        if not sibling_outputs or not any(operation.child_port(port) in step.child_inputs for port in sibling_outputs):
            return step

        child_incidence = operation.implementation.structural_analysis().incidence
        try:
            validate_request_structure(
                child_incidence,
                set(step.child_inputs),
                operation.child_port(output),
            )
        except BiltOverconstrainedError:
            reduced = _assignment_step(operation, output, self.established - sibling_outputs, self.backend)
            if reduced is not None and reduced.child_inputs is not None:
                try:
                    validate_request_structure(
                        child_incidence,
                        set(reduced.child_inputs),
                        operation.child_port(output),
                    )
                except BiltOverconstrainedError:
                    pass
                else:
                    return reduced
        return step

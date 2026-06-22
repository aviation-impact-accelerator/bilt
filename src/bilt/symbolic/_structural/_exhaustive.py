from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from itertools import product
from typing import Literal


Variable = str
ProducerKey = str
Kind = Literal["equation", "transform"]

Isolate = Callable[[object, Variable], object | None]


# =============================================================================
# Planning source
# =============================================================================


@dataclass(frozen=True)
class EquationSpec:
    """
    Equation known to the planner.

    `tree` is the original symbolic equation tree. This is what will later be
    passed to isolate(tree, output).

    `variables` should contain all variables appearing in the equation.
    """

    key: ProducerKey
    tree: object
    variables: frozenset[Variable]


@dataclass(frozen=True)
class TransformSpec:
    """
    Transform known to the planner.

    Transforms are already executable producers with fixed inputs and output.
    """

    key: ProducerKey
    transform: object
    inputs: frozenset[Variable]
    output: Variable


@dataclass(frozen=True)
class PlanningSource:
    """
    Concrete source of equations and transforms for planning.

    This replaces the earlier ambiguous `case.graph` adapter shape. The planner
    can now directly access equation trees through source.equations.
    """

    equations: tuple[EquationSpec, ...]
    transforms: tuple[TransformSpec, ...] = ()


# =============================================================================
# Type-state records
# =============================================================================


@dataclass(frozen=True)
class Case:
    source: PlanningSource
    inputs: frozenset[Variable]
    outputs: frozenset[Variable]


@dataclass(frozen=True)
class StructuralCandidate:
    """
    A possible producer-output edge in the structural graph.

    For equations:
        producer = original equation tree

    For transforms:
        producer = transform object

    This does not mean an equation has been isolated. It only means the
    structure says this producer may produce this output.
    """

    producer_key: ProducerKey
    producer: object
    kind: Kind
    output: Variable
    inputs: frozenset[Variable]


@dataclass(frozen=True)
class StructuralPlan:
    """
    Backend-selected structural plan.

    Steps are ordered structurally, but equation steps have not yet been
    lowered into isolated executable expressions.
    """

    steps: tuple[StructuralCandidate, ...]


@dataclass(frozen=True)
class RealizedStep:
    """
    A structural step with an executable implementation.

    For equations:
        implementation = isolated expression

    For transforms:
        implementation = transform object
    """

    candidate: StructuralCandidate
    implementation: object


@dataclass(frozen=True)
class ExecutionPlan:
    steps: tuple[RealizedStep, ...]


@dataclass(frozen=True)
class RealizationFailure:
    candidate: StructuralCandidate
    reason: str


@dataclass(frozen=True)
class PlanningResult:
    plan: ExecutionPlan | None
    structural_candidates: tuple[StructuralCandidate, ...]
    rejected: tuple[RealizationFailure, ...]


# =============================================================================
# Structural candidate generation
# =============================================================================


def generate_structural_candidates(case: Case) -> tuple[StructuralCandidate, ...]:
    """
    Generate structural candidates without isolating equations.

    For each equation, generate one candidate per possible output variable.
    Equation candidates whose output is already a case input are skipped.
    """

    candidates: list[StructuralCandidate] = []

    for equation in case.source.equations:
        for output in equation.variables:
            if output in case.inputs:
                continue

            candidates.append(
                StructuralCandidate(
                    producer_key=equation.key,
                    producer=equation.tree,
                    kind="equation",
                    output=output,
                    inputs=frozenset(equation.variables - {output}),
                )
            )

    for transform in case.source.transforms:
        if transform.output in case.inputs:
            continue

        candidates.append(
            StructuralCandidate(
                producer_key=transform.key,
                producer=transform.transform,
                kind="transform",
                output=transform.output,
                inputs=transform.inputs,
            )
        )

    return tuple(candidates)


# =============================================================================
# Structural helpers
# =============================================================================


def group_by_producer(
    candidates: Iterable[StructuralCandidate],
) -> dict[ProducerKey, list[StructuralCandidate]]:
    groups: dict[ProducerKey, list[StructuralCandidate]] = {}

    for candidate in candidates:
        groups.setdefault(candidate.producer_key, []).append(candidate)

    return groups


def schedule_structural_candidates(
    candidates: Iterable[StructuralCandidate],
    *,
    known_inputs: frozenset[Variable],
    targets: frozenset[Variable],
) -> StructuralPlan | None:
    """
    Try to order selected structural candidates from known inputs.

    This is still structural only. It does not isolate equations.
    """

    available = set(known_inputs)
    remaining = list(candidates)
    ordered: list[StructuralCandidate] = []

    while remaining:
        next_candidate = None

        for candidate in remaining:
            if candidate.inputs <= available:
                next_candidate = candidate
                break

        if next_candidate is None:
            return None

        ordered.append(next_candidate)
        available.add(next_candidate.output)
        remaining.remove(next_candidate)

    if not targets <= available:
        return None

    return StructuralPlan(steps=tuple(ordered))


# =============================================================================
# Exhaustive backend
# =============================================================================


PlanningBackend = Callable[
    [tuple[StructuralCandidate, ...], Case],
    Iterable[StructuralPlan],
]


def exhaustive_backend(
    candidates: tuple[StructuralCandidate, ...],
    case: Case,
) -> Iterator[StructuralPlan]:
    """
    Exhaustively enumerate structural producer-output choices.

    This backend chooses zero or one output per producer. It does not isolate
    equations. It only yields structurally valid plans.
    """

    groups = group_by_producer(candidates)

    options_by_producer: list[list[StructuralCandidate | None]] = [
        [None, *options]
        for options in groups.values()
    ]

    for choice in product(*options_by_producer):
        chosen = tuple(
            candidate
            for candidate in choice
            if candidate is not None
        )

        produced_outputs = [candidate.output for candidate in chosen]
        produced_output_set = set(produced_outputs)

        # Reject duplicate produced variables.
        if len(produced_outputs) != len(produced_output_set):
            continue

        # Requested outputs must already be inputs or produced by this choice.
        if not case.outputs <= case.inputs | produced_output_set:
            continue

        structural_plan = schedule_structural_candidates(
            chosen,
            known_inputs=case.inputs,
            targets=case.outputs,
        )

        if structural_plan is not None:
            yield structural_plan


# =============================================================================
# Realisation
# =============================================================================


def realize_step(
    candidate: StructuralCandidate,
    *,
    isolate: Isolate,
) -> RealizedStep | RealizationFailure:
    """
    Lower one selected structural candidate into an executable step.

    Equation candidates are isolated here.

    This is where the equation tree is used:

        isolate(candidate.producer, candidate.output)

    For equation candidates, candidate.producer is the original equation tree
    from EquationSpec.tree.
    """

    if candidate.kind == "equation":
        implementation = isolate(candidate.producer, candidate.output)

        if implementation is None:
            return RealizationFailure(
                candidate=candidate,
                reason=f"Could not isolate {candidate.output!r}",
            )

        return RealizedStep(
            candidate=candidate,
            implementation=implementation,
        )

    if candidate.kind == "transform":
        return RealizedStep(
            candidate=candidate,
            implementation=candidate.producer,
        )

    raise TypeError(f"Unknown candidate kind: {candidate.kind!r}")


def realize_structural_plan(
    structural_plan: StructuralPlan,
    *,
    isolate: Isolate,
) -> tuple[ExecutionPlan | None, tuple[RealizationFailure, ...]]:
    """
    Realise a structural plan.

    If any selected equation cannot be isolated, the whole structural plan is
    rejected and the caller can try the next backend proposal.
    """

    realized_steps: list[RealizedStep] = []

    for candidate in structural_plan.steps:
        result = realize_step(candidate, isolate=isolate)

        if isinstance(result, RealizationFailure):
            return None, (result,)

        realized_steps.append(result)

    return ExecutionPlan(steps=tuple(realized_steps)), tuple()


# =============================================================================
# Top-level planning
# =============================================================================


def build_execution_plan(
    *,
    case: Case,
    isolate: Isolate,
    backend: PlanningBackend = exhaustive_backend,
) -> PlanningResult:
    """
    Build an executable plan.

    Flow:
        Case
          -> StructuralCandidate pool
          -> backend yields StructuralPlan proposals
          -> selected StructuralPlan is realised using isolate()
          -> ExecutionPlan
    """

    structural_candidates = generate_structural_candidates(case)
    rejected: list[RealizationFailure] = []

    for structural_plan in backend(structural_candidates, case):
        execution_plan, failures = realize_structural_plan(
            structural_plan,
            isolate=isolate,
        )

        if execution_plan is not None:
            return PlanningResult(
                plan=execution_plan,
                structural_candidates=structural_candidates,
                rejected=tuple(rejected),
            )

        rejected.extend(failures)

    return PlanningResult(
        plan=None,
        structural_candidates=structural_candidates,
        rejected=tuple(rejected),
    )


# =============================================================================
# Small usage sketch
# =============================================================================


def _example() -> None:
    """
    This is only a sketch. Replace equation trees and isolate() with the real
    symbolic Tree and isolate implementation.
    """

    equation_tree = object()

    source = PlanningSource(
        equations=(
            EquationSpec(
                key="eq_1",
                tree=equation_tree,
                variables=frozenset({"a", "b", "c"}),
            ),
        ),
    )

    case = Case(
        source=source,
        inputs=frozenset({"b", "c"}),
        outputs=frozenset({"a"}),
    )

    def fake_isolate(tree: object, output: Variable) -> object | None:
        if tree is equation_tree and output == "a":
            return "isolated expression for a"
        return None

    result = build_execution_plan(
        case=case,
        isolate=fake_isolate,
    )

    assert result.plan is not None
    assert result.plan.steps[0].implementation == "isolated expression for a"
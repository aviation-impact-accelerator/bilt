import typing

if typing.TYPE_CHECKING:
    from collections.abc import Iterable

from bilt.symbolic._execution_planning._build import build_execution_plan
from bilt.symbolic._structural_analysis._candidates import generate_structural_candidates
from bilt.symbolic._structural_analysis._sources import (
    CandidateSource,
    EquationSpec,
    TransformSpec,
)
from bilt.symbolic._structural_analysis._types import Case, StructuralCandidate, StructuralPlan
from bilt.symbolic._structural_analysis.backends._exhaustive import exhaustive_backend


def test_generate_structural_candidates_skips_known_inputs_and_carries_producers() -> None:
    equation_tree = object()
    transform = object()

    source = CandidateSource(
        equations=(
            EquationSpec(
                key="eq_1",
                producer=equation_tree,
                variables=frozenset({"a", "b", "c"}),
            ),
        ),
        transforms=(
            TransformSpec(
                key="tr_1",
                producer=transform,
                inputs=frozenset({"a"}),
                output="d",
            ),
        ),
    )

    candidates = generate_structural_candidates(
        source,
        known_inputs=frozenset({"b"}),
    )

    by_output = {candidate.output: candidate for candidate in candidates}

    assert set(by_output) == {"a", "c", "d"}

    assert by_output["a"] == StructuralCandidate(
        producer_key="eq_1",
        kind="equation",
        producer=equation_tree,
        output="a",
        inputs=frozenset({"b", "c"}),
    )
    assert by_output["c"] == StructuralCandidate(
        producer_key="eq_1",
        kind="equation",
        producer=equation_tree,
        output="c",
        inputs=frozenset({"a", "b"}),
    )
    assert by_output["d"] == StructuralCandidate(
        producer_key="tr_1",
        kind="transform",
        producer=transform,
        output="d",
        inputs=frozenset({"a"}),
    )


def test_exhaustive_backend_yields_schedulable_plan() -> None:
    case = Case(
        inputs=frozenset({"a"}),
        outputs=frozenset({"c"}),
    )
    candidates = (
        StructuralCandidate(
            producer_key="step_1",
            kind="transform",
            producer=object(),
            output="b",
            inputs=frozenset({"a"}),
        ),
        StructuralCandidate(
            producer_key="step_2",
            kind="transform",
            producer=object(),
            output="c",
            inputs=frozenset({"b"}),
        ),
    )

    plans = tuple(exhaustive_backend(candidates, case))

    assert plans == (
        StructuralPlan(
            steps=(
                candidates[0],
                candidates[1],
            ),
        ),
    )


def test_exhaustive_backend_rejects_duplicate_outputs() -> None:
    case = Case(
        inputs=frozenset({"a"}),
        outputs=frozenset({"b"}),
    )
    candidates = (
        StructuralCandidate(
            producer_key="step_1",
            kind="transform",
            producer=object(),
            output="b",
            inputs=frozenset({"a"}),
        ),
        StructuralCandidate(
            producer_key="step_2",
            kind="transform",
            producer=object(),
            output="b",
            inputs=frozenset({"a"}),
        ),
    )

    plans = tuple(exhaustive_backend(candidates, case))
    length = 2
    assert len(plans) == length
    assert all(len(plan.steps) == 1 for plan in plans)


def test_build_execution_plan_realises_equation_and_transform_steps() -> None:
    equation_tree = object()

    def transform(inputs: dict[str, object]) -> object:
        return inputs["a"]

    source = CandidateSource(
        equations=(
            EquationSpec(
                key="eq_1",
                producer=equation_tree,
                variables=frozenset({"a", "b"}),
            ),
        ),
        transforms=(
            TransformSpec(
                key="tr_1",
                producer=transform,
                inputs=frozenset({"a"}),
                output="c",
            ),
        ),
    )
    case = Case(
        inputs=frozenset({"b"}),
        outputs=frozenset({"c"}),
    )

    def isolate(tree: object, output: str) -> object | None:
        if tree is equation_tree and output == "a":
            return "isolated a"
        return None

    result = build_execution_plan(
        source=source,
        case=case,
        isolate=isolate,
        backend=exhaustive_backend,
    )

    assert result.plan is not None
    assert [step.candidate.output for step in result.plan.steps] == ["a", "c"]
    assert result.plan.steps[0].implementation == "isolated a"
    assert result.plan.steps[1].implementation is transform


def test_build_execution_plan_records_failed_realisation_and_tries_next_plan() -> None:
    bad_tree = object()
    good_tree = object()

    source = CandidateSource(
        equations=(
            EquationSpec(
                key="bad_eq",
                producer=bad_tree,
                variables=frozenset({"x", "y"}),
            ),
            EquationSpec(
                key="good_eq",
                producer=good_tree,
                variables=frozenset({"x", "y"}),
            ),
        ),
    )
    case = Case(
        inputs=frozenset({"y"}),
        outputs=frozenset({"x"}),
    )

    def isolate(tree: object, output: str) -> object | None:
        if tree is good_tree and output == "x":
            return "isolated x"
        return None

    def deterministic_backend(
        candidates: tuple[StructuralCandidate, ...],
        case: Case,
    ) -> Iterable[StructuralPlan]:
        del case
        by_key = {candidate.producer_key: candidate for candidate in candidates}
        yield StructuralPlan(steps=(by_key["bad_eq"],))
        yield StructuralPlan(steps=(by_key["good_eq"],))

    result = build_execution_plan(
        source=source,
        case=case,
        isolate=isolate,
        backend=deterministic_backend,
    )

    assert result.plan is not None
    assert result.plan.steps[0].candidate.producer_key == "good_eq"
    assert result.plan.steps[0].implementation == "isolated x"
    assert len(result.rejected) == 1
    assert result.rejected[0].candidate.producer_key == "bad_eq"

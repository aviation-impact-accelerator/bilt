"""Backend-independent call schedule for a realised Bilt evaluation.

The schedule is produced from a selected plan before backend compilation.
It describes dependency ordering and source kinds without embedding Python
AST or native callable objects.
"""

import dataclasses
import typing

from bilt.core._port import mangle_name

if typing.TYPE_CHECKING:
    from bilt._backend._case import BackendCase
    from bilt.core._structural_analysis._plan import AnalysisPlan, PlanStep
    from bilt.core._structural_analysis._qualified_block import QualifiedPlan, QualifiedStep
    from bilt.symbolic._domains import Domain


@dataclasses.dataclass(frozen=True, slots=True)
class CallStep:
    """One named computation and the previously available values it consumes."""

    output: str
    inputs: tuple[str, ...]
    domain: Domain | None = None
    kind: typing.Literal["primitive", "equation", "transform", "child", "linear_solve", "projection"] = "primitive"
    block_index: int | None = None


@dataclasses.dataclass(frozen=True, slots=True)
class ProgramIR:
    """A directed dependency schedule, without Python AST or native callables."""

    parameters: tuple[str, ...]
    steps: tuple[CallStep, ...]
    result: str

    def verify(self) -> None:
        available = set(self.parameters)
        if len(available) != len(self.parameters):
            message = "Duplicate program parameter."
            raise ValueError(message)
        for step in self.steps:
            if step.output in available:
                message = f"Program value {step.output!r} has multiple producers."
                raise ValueError(message)
            missing = set(step.inputs) - available
            if missing:
                message = f"Program step {step.output!r} needs unavailable inputs: {sorted(missing)!r}."
                raise ValueError(message)
            available.add(step.output)
        if self.result not in available:
            message = f"Program result {self.result!r} has no producer."
            raise ValueError(message)


def program_from_case(case: BackendCase) -> ProgramIR:
    """Snapshot compiled case dependencies without embedding backend callables."""
    steps = tuple(
        CallStep(
            output=step.name,
            inputs=step.parameters,
            domain=step.output_domain,
            kind="linear_solve" if step.name.startswith("__bilt_linear_solution_") else "primitive",
        )
        for step in case.intermediates
    )
    program = ProgramIR(parameters=case.parameters, steps=steps, result=case.return_name)
    program.verify()
    return program


def _operation_kind(step: PlanStep | QualifiedStep) -> typing.Literal["equation", "transform", "child", "primitive"]:
    """Identify a model operation without importing any backend functionality."""
    from bilt.core._equation import BasicEquation  # noqa: PLC0415
    from bilt.core._sub_system import SubSystemConstituent  # noqa: PLC0415
    from bilt.core._transform import BasicTransform  # noqa: PLC0415

    source = step.operation if hasattr(step, "operation") else step.source
    if isinstance(source, BasicEquation):
        return "equation"
    if isinstance(source, BasicTransform):
        return "transform"
    if isinstance(source, SubSystemConstituent):
        return "child"
    return "primitive"


def program_from_plan(plan: AnalysisPlan) -> ProgramIR:
    """Describe direct steps and coupled solves *before* any backend compilation.

    A failed candidate can have unresolved dependencies; verification of the
    complete graph is deferred until the historical realisation failure path
    has had the opportunity to reject the candidate.
    """
    reserved = {mangle_name(step.output) for step in plan.steps} | set(map(mangle_name, plan.inputs))
    steps: list[CallStep] = []
    for index, block in enumerate(plan.blocks):
        if block.is_cyclic:
            solution_name = f"__bilt_linear_solution_{index}"
            while solution_name in reserved:
                solution_name += "_"
            reserved.add(solution_name)
            steps.append(
                CallStep(
                    solution_name,
                    tuple(map(mangle_name, sorted(block.inputs))),
                    kind="linear_solve",
                    block_index=index,
                )
            )
            steps.extend(
                CallStep(mangle_name(output), (solution_name,), kind="projection", block_index=index)
                for output in sorted(block.outputs)
            )
            continue

        step = block.steps[0]
        # Keep the legacy unsupported-constituent error in the compiler rather
        # than raising it while describing an otherwise unexecutable plan.
        try:
            inputs = step.consumed
        except NotImplementedError:
            inputs = ()
        steps.append(
            CallStep(
                mangle_name(step.output),
                tuple(map(mangle_name, inputs)),
                kind=_operation_kind(step),
                block_index=index,
            )
        )
    return ProgramIR(
        parameters=tuple(map(mangle_name, sorted(plan.inputs))),
        steps=tuple(steps),
        result=mangle_name(plan.output),
    )


def program_from_qualified_plan(plan: QualifiedPlan) -> ProgramIR:
    """Describe qualified primitive routes and grouped solves before transpiling."""
    from bilt.core._structural_analysis._qualified_block import QualifiedBlock  # noqa: PLC0415

    reserved = set(map(mangle_name, plan.outputs | plan.request.input_domains.keys()))
    steps: list[CallStep] = []
    for index, block in enumerate(plan.blocks):
        if isinstance(block, QualifiedBlock):
            solution_name = f"__bilt_qualified_solution_{index}"
            while solution_name in reserved:
                solution_name += "_"
            reserved.add(solution_name)
            steps.append(
                CallStep(
                    solution_name,
                    tuple(map(mangle_name, sorted(block.inputs))),
                    kind="linear_solve",
                    block_index=index,
                )
            )
            steps.extend(
                CallStep(mangle_name(output), (solution_name,), kind="projection", block_index=index)
                for output in block.unknowns
            )
            continue
        steps.append(
            CallStep(
                mangle_name(block.output),
                tuple(map(mangle_name, sorted(block.inputs))),
                kind=_operation_kind(block),
                block_index=index,
            )
        )
    return ProgramIR(
        parameters=tuple(map(mangle_name, sorted(plan.request.input_domains))),
        steps=tuple(steps),
        result=mangle_name(plan.request.output),
    )

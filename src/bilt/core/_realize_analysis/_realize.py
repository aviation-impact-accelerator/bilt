import ast
import dataclasses
import inspect
import textwrap
import typing

from bilt._backend._case import BackendCase
from bilt.core._equation import BasicEquation
from bilt.core._system._interface import Interface
from bilt.core._transform import BasicTransform
from bilt.symbolic._isolation._errors import IsolationError
from bilt.symbolic._isolation._isolate import equation_to_ast

if typing.TYPE_CHECKING:
    from bilt._backend._base_backend import Backend
    from bilt._backend._callable import BackendCallable
    from bilt.core._structural_analysis._plan import AnalysisPlan, PlanStep


def realize_plan(plan: AnalysisPlan, backend: Backend) -> BackendCase | RealizationFailure:
    """Convert an AnalysisPlan into a BackendCase by compiling each step to a callable."""
    intermediates = []
    for step in plan.steps:
        result = _operation_to_callable(step, backend=backend)
        if isinstance(result, RealizationFailure):
            return result
        intermediates.append(result)

    return BackendCase(
        name=plan.output,
        parameters=tuple(sorted(plan.inputs)),
        intermediates=intermediates,
        return_name=plan.output,
    )


def _operation_to_callable(plan_step: PlanStep, backend: Backend) -> BackendCallable | RealizationFailure:
    operation = plan_step.operation
    output = plan_step.output
    match operation:
        case BasicTransform():
            function_ast = typing.cast(
                "ast.FunctionDef",
                ast.parse(textwrap.dedent(inspect.getsource(operation._function))).body[0],  # noqa: SLF001
            )
            function_ast.decorator_list = [
                d for d in function_ast.decorator_list if not (isinstance(d, ast.Name) and d.id == "transform")
            ]
            return backend.transpile(function_ast)

        case BasicEquation():
            try:
                function_ast = equation_to_ast(operation, output)
                return backend.transpile(function_ast)
            except IsolationError:
                return RealizationFailure(failed_plan_step=plan_step)
        case Interface():
            msg = "Interface evaluation is not yet implemented."
            raise NotImplementedError(msg)
        case _:
            msg = f"Operation type {type(operation)} is not supported for evaluation."
            raise NotImplementedError(msg)


@dataclasses.dataclass
class RealizationFailure:
    failed_plan_step: PlanStep

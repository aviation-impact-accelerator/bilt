__all__ = ["RealizationFailure", "realize_plan"]

import ast
import dataclasses
import inspect
import textwrap
import typing

from bilt._backend._case import BackendCase
from bilt._backend._errors import BiltBackendError
from bilt.core._equation import BasicEquation
from bilt.core._interface import Interface
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
    match plan_step.operation:
        case BasicTransform() as transform:
            function_ast = typing.cast(
                "ast.FunctionDef",
                ast.parse(textwrap.dedent(inspect.getsource(transform._function))).body[0],  # noqa: SLF001
            )
            function_ast.decorator_list = [
                d for d in function_ast.decorator_list if not (isinstance(d, ast.Name) and d.id == "transform")
            ]
            return _guarded_transpile(function_ast, plan_step, backend)

        case BasicEquation() as equation:
            try:
                function_ast = equation_to_ast(equation, plan_step.output)
            except IsolationError as error:
                return RealizationFailure(failed_plan_step=plan_step, cause=error)
            return _guarded_transpile(function_ast, plan_step, backend)
        case Interface():
            msg = "Interface evaluation is not yet implemented."
            raise NotImplementedError(msg)
        case _:
            msg = f"Operation type {type(plan_step.operation)} is not supported for evaluation."
            raise NotImplementedError(msg)


def _guarded_transpile(
    function_ast: ast.FunctionDef, plan_step: PlanStep, backend: Backend
) -> BackendCallable | RealizationFailure:
    try:
        return backend.transpile(function_ast)
    except BiltBackendError as error:
        return RealizationFailure(failed_plan_step=plan_step, permanent=False, cause=error)


@dataclasses.dataclass
class RealizationFailure:
    """A plan step that could not be compiled to a callable.

    ``permanent`` distinguishes a step that can never produce its output -- an equation that
    cannot be isolated for the requested parameter -- from one that merely failed for this
    query's inputs or backend.

    ``cause`` is the underlying exception, kept so a search that exhausts every route can report
    why the routes it tried were rejected instead of only that none were left.
    """

    failed_plan_step: PlanStep
    permanent: bool = True
    cause: Exception | None = None

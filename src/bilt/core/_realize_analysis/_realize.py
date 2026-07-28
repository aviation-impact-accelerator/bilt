import ast
import inspect
import textwrap
import typing

from bilt._backend._case import BackendCase
from bilt.core._equation import BasicEquation
from bilt.core._system._interface import Interface
from bilt.core._transform import BasicTransform
from bilt.symbolic._isolation._isolate import equation_to_ast

if typing.TYPE_CHECKING:
    from bilt._backend._base_backend import Backend
    from bilt._backend._callable import BackendCallable
    from bilt.core._model_constituent import BiltModelConstituent
    from bilt.core._structural_analysis._plan import AnalysisPlan


def realize_plan(plan: AnalysisPlan, backend: Backend) -> BackendCase:
    """Convert an AnalysisPlan into a BackendCase by compiling each step to a callable."""
    return BackendCase(
        name=plan.output,
        parameters=tuple(sorted(plan.inputs)),
        intermediates=[
            _operation_to_callable(step.operation, output=step.output, backend=backend) for step in plan.steps
        ],
        return_name=plan.output,
    )


def _operation_to_callable(operation: BiltModelConstituent, *, output: str, backend: Backend) -> BackendCallable:
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
            function_ast = equation_to_ast(operation, output)
            return backend.transpile(function_ast)
        case Interface():
            msg = "Interface evaluation is not yet implemented."
            raise NotImplementedError(msg)
        case _:
            msg = f"Operation type {type(operation)} is not supported for evaluation."
            raise NotImplementedError(msg)

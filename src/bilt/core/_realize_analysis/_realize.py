__all__ = ["RealizationFailure", "case_to_callable", "mangle_name", "realize_plan"]

import ast
import dataclasses
import inspect
import textwrap
import typing

from bilt._backend._callable import BackendCallable
from bilt._backend._case import BackendCase
from bilt._backend._errors import BiltBackendError
from bilt.core._equation import BasicEquation
from bilt.core._port import DOTTED_SEPARATOR
from bilt.core._structural_analysis._errors import BiltSystemAnalysisError
from bilt.core._sub_system import SubSystemConstituent
from bilt.core._transform import BasicTransform
from bilt.symbolic._isolation._errors import IsolationError
from bilt.symbolic._isolation._isolate import equation_to_ast

if typing.TYPE_CHECKING:
    from bilt._backend._base_backend import Backend
    from bilt.core._evaluation_request import EvaluationRequest
    from bilt.core._structural_analysis._plan import AnalysisPlan, PlanStep

MANGLED_SEPARATOR: typing.Final = "__"


def mangle_name(name: str) -> str:
    return name.replace(DOTTED_SEPARATOR, MANGLED_SEPARATOR)


def case_to_callable(case: BackendCase, request: EvaluationRequest, backend: Backend) -> BackendCallable:
    """Chain a case into one callable, answering ``request`` in the names it was asked in."""
    chained = backend.chain_case(case)
    canonical_by_mangled = {mangle_name(name): name for name in request.available}
    return BackendCallable(
        name=chained.name,
        parameters=tuple(canonical_by_mangled[name] for name in chained.parameters),
        callable=chained.callable,
    )


def realize_plan(plan: AnalysisPlan, backend: Backend) -> BackendCase | RealizationFailure:
    """Convert an AnalysisPlan into a BackendCase by compiling each step to a callable."""
    intermediates = []
    for step in plan.steps:
        result = _operation_to_callable(step, backend=backend)
        if isinstance(result, RealizationFailure):
            return result
        intermediates.append(result)

    output = mangle_name(plan.output)
    return BackendCase(
        name=output,
        parameters=tuple(map(mangle_name, sorted(plan.inputs))),
        intermediates=intermediates,
        return_name=output,
    )


def _operation_to_callable(plan_step: PlanStep, backend: Backend) -> BackendCallable | RealizationFailure:
    match plan_step.operation:
        case BasicTransform() as transform:
            return _transform_to_callable(transform, plan_step, backend)
        case BasicEquation() as equation:
            return _equation_to_callable(equation, plan_step, backend)
        case SubSystemConstituent() as sub_system:
            return _sub_system_to_callable(sub_system, plan_step, backend)
        case _:
            msg = f"Operation type {type(plan_step.operation)} is not supported for evaluation."
            raise NotImplementedError(msg)


def _transform_to_callable(
    transform: BasicTransform,
    plan_step: PlanStep,
    backend: Backend,
) -> BackendCallable | RealizationFailure:
    """Realize a transform step by recovering its source as an AST, with the ``@transform`` decorator stripped."""
    function_ast = typing.cast(
        "ast.FunctionDef",
        ast.parse(textwrap.dedent(inspect.getsource(transform._function))).body[0],  # noqa: SLF001
    )
    function_ast.decorator_list = [
        d for d in function_ast.decorator_list if not (isinstance(d, ast.Name) and d.id == "transform")
    ]
    return _guarded_transpile(function_ast, plan_step, backend)


def _equation_to_callable(
    equation: BasicEquation,
    plan_step: PlanStep,
    backend: Backend,
) -> BackendCallable | RealizationFailure:
    """Realize an equation step by isolating it for the step's output, then transpiling that AST."""
    try:
        function_ast = equation_to_ast(equation, plan_step.output)
    except IsolationError as error:
        return RealizationFailure(failed_plan_step=plan_step, cause=error)
    return _guarded_transpile(function_ast, plan_step, backend)


def _sub_system_to_callable(
    sub_system: SubSystemConstituent,
    plan_step: PlanStep,
    backend: Backend,
) -> BackendCallable | RealizationFailure:
    """Realize a sub-system step by splicing the child's own compiled callable into the parent."""
    route = sub_system.child_route(plan_step.output)
    if route is None:
        msg = f"{sub_system!r} has no internal route to port {plan_step.output!r} from its other ports."
        return RealizationFailure(failed_plan_step=plan_step, permanent=False, cause=BiltSystemAnalysisError(msg))

    try:
        child_callable = sub_system.implementation.get_evaluation(route, backend)
    except BiltSystemAnalysisError as error:
        return RealizationFailure(failed_plan_step=plan_step, permanent=False, cause=error)

    return BackendCallable(
        name=mangle_name(plan_step.output),
        parameters=tuple(mangle_name(sub_system.qualify(port)) for port in child_callable.parameters),
        callable=child_callable.callable,
    )


def _guarded_transpile(
    function_ast: ast.FunctionDef, plan_step: PlanStep, backend: Backend
) -> BackendCallable | RealizationFailure:
    """Transpile an AST, turning a backend's refusal into a failure the search can route around."""
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

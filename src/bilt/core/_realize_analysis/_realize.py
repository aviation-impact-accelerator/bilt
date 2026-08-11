"""Compile an analysis plan into a backend case.

Everything the plan names is dotted; everything the compiled case names is mangled, because
generated code has to spell names as Python identifiers. This module is that boundary.
"""

__all__ = ["RealizationFailure", "canonicalize_backend_callable", "realize_plan"]

import ast
import dataclasses
import inspect
import textwrap
import typing

from bilt._backend._callable import BackendCallable
from bilt._backend._case import BackendCase
from bilt.core._equation import BasicEquation
from bilt.core._port import DOTTED_SEPARATOR
from bilt.core._structural_analysis._errors import BiltSystemAnalysisError
from bilt.core._sub_system import SubSystemConstituent
from bilt.core._transform import BasicTransform
from bilt.symbolic._isolation._errors import IsolationError
from bilt.symbolic._isolation._isolate import equation_to_ast

if typing.TYPE_CHECKING:
    from bilt._backend._base_backend import Backend
    from bilt.core._structural_analysis._plan import AnalysisPlan, PlanStep

MANGLED_SEPARATOR: typing.Final = "__"


def mangle_name(name: str) -> str:
    """Return the Python-identifier spelling of any plan name.

    Total on plan names: a plain name such as ``v`` mangles to itself, while a qualified
    name such as ``resistor.v`` mangles to ``resistor__v``.
    """
    return name.replace(DOTTED_SEPARATOR, MANGLED_SEPARATOR)


def canonicalize_backend_callable(
    mangled_callable: BackendCallable,
    canonical_parameters: tuple[str, ...],
) -> BackendCallable:
    """Wrap a callable that speaks mangled parameter names so it accepts canonical ones."""
    if not any(DOTTED_SEPARATOR in name for name in canonical_parameters):
        # No name is dotted, so the mangled spelling already is the canonical one.
        return dataclasses.replace(mangled_callable, parameters=canonical_parameters)

    canonical_to_mangled = {name: mangle_name(name) for name in canonical_parameters}
    inner = mangled_callable.callable

    def call_with_canonical_names(*args: object, **kwargs: object) -> object:
        named = dict(zip(canonical_parameters, args, strict=False)) | kwargs
        return inner(**{canonical_to_mangled[name]: value for name, value in named.items()})

    return dataclasses.replace(
        mangled_callable,
        parameters=canonical_parameters,
        callable=call_with_canonical_names,
    )


class _NameMangler(ast.NodeTransformer):
    """Rewrite dotted symbol names into valid Python identifiers, in signatures and bodies alike."""

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        node.name = mangle_name(node.name)
        self.generic_visit(node)
        return node

    def visit_Name(self, node: ast.Name) -> ast.Name:
        node.id = mangle_name(node.id)
        return node

    def visit_arg(self, node: ast.arg) -> ast.arg:
        node.arg = mangle_name(node.arg)
        return node


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
        parameters=tuple(mangle_name(name) for name in sorted(plan.inputs)),
        intermediates=intermediates,
        return_name=output,
    )


def _operation_to_callable(plan_step: PlanStep, backend: Backend) -> BackendCallable | RealizationFailure:
    match plan_step.operation:
        case BasicTransform() as transform:
            return backend.transpile(_transform_to_ast(transform))

        case BasicEquation() as equation:
            try:
                function_ast = equation_to_ast(equation, plan_step.output)
            except IsolationError:
                return RealizationFailure(failed_plan_step=plan_step)
            return backend.transpile(_NameMangler().visit(function_ast))

        case SubSystemConstituent() as sub_system:
            return _sub_system_to_callable(sub_system, plan_step, backend)

        case _:
            msg = f"Operation type {type(plan_step.operation)} is not supported for evaluation."
            raise NotImplementedError(msg)


def _transform_to_ast(transform: BasicTransform) -> ast.FunctionDef:
    """Recover a transform's source as an AST, with the ``@transform`` decorator stripped."""
    function_ast = typing.cast(
        "ast.FunctionDef",
        ast.parse(textwrap.dedent(inspect.getsource(transform._function))).body[0],  # noqa: SLF001
    )
    function_ast.decorator_list = [
        d for d in function_ast.decorator_list if not (isinstance(d, ast.Name) and d.id == "transform")
    ]
    return function_ast


def _sub_system_to_callable(
    sub_system: SubSystemConstituent,
    plan_step: PlanStep,
    backend: Backend,
) -> BackendCallable | RealizationFailure:
    """Realize a sub-system step by splicing in the child's own compiled callable."""
    child_output = sub_system.unqualify(plan_step.output)
    if child_output is None:
        msg = f"{sub_system!r} does not have port {plan_step.output!r}."
        raise BiltSystemAnalysisError(msg)

    available_child_ports = set(sub_system.interface.__bilt_parameters__) - {child_output}
    child_ports = sub_system.implementation.analyser.required_inputs_for_target(available_child_ports, child_output)
    if child_ports is None:
        return RealizationFailure(failed_plan_step=plan_step)

    try:
        child_callable = sub_system.implementation.realized_callable(child_ports, child_output, backend)
    except BiltSystemAnalysisError:
        return RealizationFailure(failed_plan_step=plan_step)

    return BackendCallable(
        name=mangle_name(plan_step.output),
        parameters=tuple(mangle_name(sub_system.port(port).dotted) for port in child_callable.parameters),
        callable=child_callable.callable,
    )


@dataclasses.dataclass(frozen=True)
class RealizationFailure:
    failed_plan_step: PlanStep

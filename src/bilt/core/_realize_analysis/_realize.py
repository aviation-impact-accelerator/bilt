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
from bilt.core._structural_analysis._plan import AnalysisPlan, CoupledBlock, PlanStep
from bilt.core._structural_analysis._scc._coupled import analyse_coupled_block
from bilt.core._sub_system import SubSystemConstituent
from bilt.core._transform import BasicTransform
from bilt.symbolic._atoms import (
    ADD,
    INV,
    MUL,
    NEG,
    POW,
    FloatType,
    IntType,
    SymbolType,
)
from bilt.symbolic._isolation._errors import IsolationError
from bilt.symbolic._isolation._isolate import equation_to_ast
from bilt.symbolic._tree import Tree

if typing.TYPE_CHECKING:
    from bilt._backend._base_backend import Backend

MANGLED_SEPARATOR: typing.Final = "__"


def mangle_name(name: str) -> str:
    """Return the Python-identifier spelling of any plan name."""
    return name.replace(DOTTED_SEPARATOR, MANGLED_SEPARATOR)


def canonicalize_backend_callable(
    mangled_callable: BackendCallable,
    canonical_parameters: tuple[str, ...],
) -> BackendCallable:
    """Wrap a callable that speaks mangled parameter names so it accepts canonical ones."""
    if not any(DOTTED_SEPARATOR in name for name in canonical_parameters):
        return dataclasses.replace(
            mangled_callable,
            parameters=canonical_parameters,
        )

    canonical_to_mangled = {name: mangle_name(name) for name in canonical_parameters}
    inner = mangled_callable.callable

    def call_with_canonical_names(
        *args: object,
        **kwargs: object,
    ) -> object:
        named = dict(zip(canonical_parameters, args, strict=False)) | kwargs

        return inner(**{canonical_to_mangled[name]: value for name, value in named.items()})

    return dataclasses.replace(
        mangled_callable,
        parameters=canonical_parameters,
        callable=call_with_canonical_names,
    )


class _NameMangler(ast.NodeTransformer):
    """Rewrite dotted symbol names into valid Python identifiers."""

    def visit_FunctionDef(
        self,
        node: ast.FunctionDef,
    ) -> ast.FunctionDef:
        node.name = mangle_name(node.name)
        self.generic_visit(node)
        return node

    def visit_Name(
        self,
        node: ast.Name,
    ) -> ast.Name:
        node.id = mangle_name(node.id)
        return node

    def visit_arg(
        self,
        node: ast.arg,
    ) -> ast.arg:
        node.arg = mangle_name(node.arg)
        return node


def realize_plan(
    plan: AnalysisPlan,
    backend: Backend,
) -> BackendCase | RealizationFailure:
    """Convert an AnalysisPlan into a BackendCase."""
    intermediates: list[BackendCallable] = []

    for item in plan.steps:
        match item:
            case PlanStep() as plan_step:
                result = _operation_to_callable(
                    plan_step,
                    backend,
                )

                if isinstance(result, RealizationFailure):
                    return result

                intermediates.append(result)

            case CoupledBlock() as block:
                intermediates.extend(
                    _coupled_block_to_callables(
                        block,
                        backend,
                    )
                )

    output = mangle_name(plan.output)

    return BackendCase(
        name=output,
        parameters=tuple(mangle_name(name) for name in sorted(plan.inputs)),
        intermediates=intermediates,
        return_name=output,
    )


def _operation_to_callable(
    plan_step: PlanStep,
    backend: Backend,
) -> BackendCallable | RealizationFailure:
    match plan_step.operation:
        case BasicTransform() as transform:
            return backend.transpile(_transform_to_ast(transform))

        case BasicEquation() as equation:
            try:
                function_ast = equation_to_ast(
                    equation,
                    plan_step.output,
                )
            except IsolationError:
                return RealizationFailure(failed_plan_step=plan_step)

            return backend.transpile(_NameMangler().visit(function_ast))

        case SubSystemConstituent() as sub_system:
            return _sub_system_to_callable(
                sub_system,
                plan_step,
                backend,
            )

        case _:
            msg = f"Operation type {type(plan_step.operation)} is not supported for evaluation."
            raise NotImplementedError(msg)


def _coupled_block_to_callables(
    block: CoupledBlock,
    backend: Backend,
) -> tuple[BackendCallable, ...]:
    """Realize a coupled block as one solve followed by output extraction."""
    system = analyse_coupled_block(block)

    solve_name = "__bilt_solve_" + "_".join(mangle_name(name) for name in system.unknowns)

    def solve(*args: object) -> object:
        values = dict(
            zip(
                system.parameters,
                args,
                strict=True,
            )
        )

        coefficients = [[_evaluate_tree(tree, values) for tree in row] for row in system.coefficients]

        rhs = [_evaluate_tree(tree, values) for tree in system.rhs]

        return backend.linear_solve(
            coefficients,
            rhs,
        )

    callables: list[BackendCallable] = [
        BackendCallable(
            name=solve_name,
            parameters=tuple(mangle_name(name) for name in system.parameters),
            callable=solve,
        )
    ]

    for index, unknown in enumerate(system.unknowns):

        def extract(
            solution: object,
            *,
            index: int = index,
        ) -> object:
            return solution[index]  # type: ignore[index]

        callables.append(
            BackendCallable(
                name=mangle_name(unknown),
                parameters=(solve_name,),
                callable=extract,
            )
        )

    return tuple(callables)


def _evaluate_tree(
    tree: Tree,
    values: dict[str, object],
) -> typing.Any:
    """Evaluate a symbolic expression using concrete parameter values."""
    if tree.is_leaf:
        if tree.head.atom_type is SymbolType:
            return values[typing.cast("str", tree.head.value)]

        if tree.head.atom_type in (IntType, FloatType):
            return tree.head.value

        msg = f"Unsupported symbolic leaf {tree.head!r}."
        raise BiltSystemAnalysisError(msg)

    arguments = tuple(_evaluate_tree(child, values) for child in tree.args)

    match tree.head:
        case head if head is ADD:
            return _add_values(arguments)

        case head if head is MUL:
            return _multiply_values(arguments)

        case head if head is NEG:
            return -arguments[0]

        case head if head is INV:
            return 1 / arguments[0]

        case head if head is POW:
            return arguments[0] ** arguments[1]

        case _:
            msg = f"Unsupported symbolic operation {tree.head!r}."
            raise BiltSystemAnalysisError(msg)


def _add_values(
    values: tuple[typing.Any, ...],
) -> typing.Any:
    if not values:
        msg = "Cannot evaluate an empty symbolic addition."
        raise BiltSystemAnalysisError(msg)

    result = values[0]

    for value in values[1:]:
        result = result + value

    return result


def _multiply_values(
    values: tuple[typing.Any, ...],
) -> typing.Any:
    if not values:
        msg = "Cannot evaluate an empty symbolic multiplication."
        raise BiltSystemAnalysisError(msg)

    result = values[0]

    for value in values[1:]:
        result = result * value

    return result


def _transform_to_ast(
    transform: BasicTransform,
) -> ast.FunctionDef:
    """Recover a transform's source as an AST, with the @transform decorator stripped."""
    function_ast = typing.cast(
        "ast.FunctionDef",
        ast.parse(
            textwrap.dedent(
                inspect.getsource(transform._function)  # noqa: SLF001
            )
        ).body[0],
    )

    function_ast.decorator_list = [
        decorator
        for decorator in function_ast.decorator_list
        if not (isinstance(decorator, ast.Name) and decorator.id == "transform")
    ]

    return function_ast


def _sub_system_to_callable(
    sub_system: SubSystemConstituent,
    plan_step: PlanStep,
    backend: Backend,
) -> BackendCallable | RealizationFailure:
    """Realize a sub-system step using its selected structural route."""
    child_output = sub_system.unqualify(plan_step.output)

    if child_output is None:
        msg = f"{sub_system!r} does not have port {plan_step.output!r}."
        raise BiltSystemAnalysisError(msg)

    sub_system_plan = plan_step.sub_system_plan

    if sub_system_plan is None:
        return RealizationFailure(failed_plan_step=plan_step)

    child_ports: set[str] = set()

    for qualified_port in sub_system_plan.inputs:
        child_port = sub_system.unqualify(qualified_port)

        if child_port is None:
            msg = f"Selected child input {qualified_port!r} does not belong to {sub_system!r}."
            raise BiltSystemAnalysisError(msg)

        child_ports.add(child_port)

    try:
        child_callable = sub_system.implementation.realized_callable(
            frozenset(child_ports),
            child_output,
            backend,
        )
    except BiltSystemAnalysisError:
        return RealizationFailure(failed_plan_step=plan_step)

    return BackendCallable(
        name=mangle_name(plan_step.output),
        parameters=tuple(mangle_name(sub_system.qualify(port)) for port in child_callable.parameters),
        callable=child_callable.callable,
    )


@dataclasses.dataclass(frozen=True)
class RealizationFailure:
    failed_plan_step: PlanStep

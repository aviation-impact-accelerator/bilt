import ast
import types
import typing

from bilt.core._equation_ast import BiltEquationSyntaxError, extract_equation_ast, normalise_equation_ast
from bilt.symbolic._expressions import conditional_expr

if typing.TYPE_CHECKING:
    import collections.abc


def validated_symbolic_function(
    function: collections.abc.Callable[..., object],
) -> collections.abc.Callable[..., object]:
    if not isinstance(function, types.FunctionType):
        msg = "Equations require an inspectable Python function."
        raise BiltEquationSyntaxError(msg)

    source = extract_equation_ast(function)
    has_conditional = any(isinstance(statement, ast.If) for statement in source.body)
    validated = normalise_equation_ast(source)

    if not has_conditional:
        return function
    return conditional_symbolic_function(function, validated)


def conditional_symbolic_function(
    function: types.FunctionType,
    node: ast.FunctionDef,
) -> collections.abc.Callable[..., object]:
    if function.__code__.co_freevars:
        names = ", ".join(function.__code__.co_freevars)
        msg = f"Conditional equation {function.__name__!r} captures closure variables: {names}."
        raise BiltEquationSyntaxError(msg)

    # These are already resolved from the original signature. Do not execute
    # annotations or defaults when reconstructing the temporary tracing callable.
    node.returns = None
    node.args.defaults = []
    node.args.kw_defaults = [None] * len(node.args.kwonlyargs)
    arguments = (
        *node.args.posonlyargs,
        *node.args.args,
        *node.args.kwonlyargs,
    )
    for argument in arguments:
        argument.annotation = None
    if node.args.vararg is not None:
        node.args.vararg.annotation = None
    if node.args.kwarg is not None:
        node.args.kwarg.annotation = None

    module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    namespace = dict(function.__globals__)
    namespace["_bilt_conditional"] = conditional_expr
    exec(compile(module, function.__code__.co_filename, "exec"), namespace)  # noqa: S102
    return typing.cast("collections.abc.Callable[..., object]", namespace[node.name])

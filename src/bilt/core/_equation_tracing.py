import ast
import inspect
import types
import typing

from bilt.core._equation_ast import BiltEquationSyntaxError, extract_equation_ast, normalise_equation_ast
from bilt.symbolic._expressions import conditional_expr

if typing.TYPE_CHECKING:
    import collections.abc


def conditional_symbolic_function(
    function: collections.abc.Callable[..., object],
) -> collections.abc.Callable[..., object]:
    if not isinstance(function, types.FunctionType):
        msg = "Conditional equations require a Python function."
        raise BiltEquationSyntaxError(msg)
    if function.__code__.co_freevars:
        names = ", ".join(function.__code__.co_freevars)
        msg = f"Conditional equation {function.__name__!r} captures closure variables: {names}."
        raise BiltEquationSyntaxError(msg)

    node = normalise_equation_ast(extract_equation_ast(function))

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

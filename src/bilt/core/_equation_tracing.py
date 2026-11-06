import ast
import dis
import types
import typing

from bilt.core._equation_ast import BiltEquationSyntaxError, extract_equation_ast, normalise_equation_ast
from bilt.symbolic._expressions import conditional_expr

if typing.TYPE_CHECKING:
    import collections.abc


def select_symbolic_tracing_function(
    function: collections.abc.Callable[..., object],
) -> collections.abc.Callable[..., object]:
    if not isinstance(function, types.FunctionType):
        return function
    if not any("JUMP_IF" in instruction.opname for instruction in dis.get_instructions(function)):
        return function

    try:
        node = extract_equation_ast(function)
    except BiltEquationSyntaxError:
        return function

    if not _has_symbolic_conditional(node):
        return function
    return conditional_symbolic_function(function, node)


def _has_symbolic_conditional(node: ast.FunctionDef) -> bool:
    symbols = {arg.arg for arg in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs)}
    for statement in node.body:
        if isinstance(statement, ast.If):
            predicate = statement.test
            if (
                isinstance(predicate, ast.Compare)
                and len(predicate.ops) == 1
                and isinstance(predicate.ops[0], (ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE))
                and any(
                    isinstance(name, ast.Name) and isinstance(name.ctx, ast.Load) and name.id in symbols
                    for name in ast.walk(predicate)
                )
            ):
                return True
        elif isinstance(statement, ast.Assign):
            symbols.update(target.id for target in statement.targets if isinstance(target, ast.Name))
        elif isinstance(statement, ast.AnnAssign) and isinstance(statement.target, ast.Name):
            symbols.add(statement.target.id)
    return False


def conditional_symbolic_function(
    function: types.FunctionType,
    node: ast.FunctionDef,
) -> collections.abc.Callable[..., object]:
    if function.__code__.co_freevars:
        names = ", ".join(function.__code__.co_freevars)
        msg = f"Conditional equation {function.__name__!r} captures closure variables: {names}."
        raise BiltEquationSyntaxError(msg)

    node = normalise_equation_ast(node)

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

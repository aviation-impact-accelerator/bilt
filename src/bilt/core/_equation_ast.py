import ast
import copy
import inspect
import textwrap
import typing

from bilt.core._errors import BiltDefinitionError

if typing.TYPE_CHECKING:
    import collections.abc


class BiltEquationSyntaxError(BiltDefinitionError, SyntaxError):
    """A proposed equation contains unsupported statement-level syntax."""


def extract_equation_ast(function: collections.abc.Callable[..., object]) -> ast.FunctionDef:
    """Extract a detached function AST, leaving the original callable unchanged.

    This is opt-in: existing @equation definitions do not depend on inspect.getsource.
    """
    name = getattr(function, "__name__", type(function).__name__)
    try:
        source = textwrap.dedent(inspect.getsource(function))
    except (OSError, TypeError) as error:
        msg = f"Cannot inspect source for equation {name!r}."
        raise BiltEquationSyntaxError(msg) from error

    module = ast.parse(source)
    node = next((item for item in module.body if isinstance(item, ast.FunctionDef)), None)
    if node is None or node.name != name:
        msg = f"Cannot locate function definition for equation {name!r}."
        raise BiltEquationSyntaxError(msg)

    # The @equation decorator must not be reapplied if a future phase compiles the AST.
    node.decorator_list = []
    return node


def normalise_equation_ast(function: ast.FunctionDef) -> ast.FunctionDef:
    """Validate and rewrite the supported single-if/single-assignment subset.

    Input and output are Python ASTs. The placeholder ``_bilt_conditional``
    is deliberately unresolved in this milestone; it will be linked to a
    symbolic operation in a subsequent PR.
    """
    node = copy.deepcopy(function)
    node.decorator_list = []
    _validate_function(node)

    normalised: list[ast.stmt] = []
    for statement in node.body:
        if not isinstance(statement, ast.If):
            normalised.append(statement)
            continue

        true_assignment = typing.cast("ast.Assign | ast.AnnAssign", statement.body[0])
        false_assignment = typing.cast("ast.Assign | ast.AnnAssign", statement.orelse[0])
        # A branch-local annotation is a declaration, not part of the symbolic
        # expression. The merged assignment is deliberately unannotated.
        replacement = ast.Assign(
            targets=[ast.copy_location(ast.Name(id=_assignment_name(true_assignment), ctx=ast.Store()), statement)],
            value=ast.Call(
                func=ast.Name(id="_bilt_conditional", ctx=ast.Load()),
                args=[statement.test, _assignment_value(true_assignment), _assignment_value(false_assignment)],
                keywords=[],
            ),
            type_comment=None,
        )
        normalised.append(ast.copy_location(replacement, statement))

    node.body = normalised
    return ast.fix_missing_locations(node)


def _reject(node: ast.AST, message: str) -> typing.NoReturn:
    line = getattr(node, "lineno", None)
    location = f" (line {line})" if line is not None else ""
    msg = f"Unsupported equation syntax{location}: {message}"
    raise BiltEquationSyntaxError(msg)


def _assignment_name(statement: ast.stmt) -> str:
    if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
        target = statement.targets[0]
    elif isinstance(statement, ast.AnnAssign) and statement.value is not None:
        target = statement.target
    else:
        _reject(statement, "expected one simple assignment")
    if not isinstance(target, ast.Name):
        _reject(target, "assignment target must be a local name")
    return target.id


def _assignment_value(statement: ast.stmt) -> ast.expr:
    if isinstance(statement, (ast.Assign, ast.AnnAssign)) and statement.value is not None:
        return statement.value
    _reject(statement, "expected an assignment with a value")


def _validate_expression(expression: ast.expr) -> None:
    forbidden = (
        ast.NamedExpr,
        ast.Yield,
        ast.YieldFrom,
        ast.Await,
        ast.Lambda,
        ast.ListComp,
        ast.SetComp,
        ast.DictComp,
        ast.GeneratorExp,
    )
    for node in ast.walk(expression):
        if isinstance(node, forbidden):
            _reject(node, f"{type(node).__name__} is not supported")


def _check_fresh_binding(node: ast.AST, name: str, params: set[str], assigned: set[str]) -> None:
    """Reject parameter updates and repeat bindings on a single execution path."""
    if name in params:
        _reject(node, f"cannot reassign equation parameter {name!r}")
    if name in assigned:
        _reject(node, f"local variable {name!r} is assigned more than once")


def _validate_return(function: ast.FunctionDef) -> None:
    """Require one final return of a single relation-shaped expression."""
    if not function.body or not isinstance(function.body[-1], ast.Return):
        _reject(function, "exactly one return is required at the end")
    if sum(isinstance(item, ast.Return) for item in ast.walk(function)) != 1:
        _reject(function, "exactly one return is required")

    final_return = function.body[-1]
    if final_return.value is None or isinstance(final_return.value, (ast.Tuple, ast.List)):
        _reject(final_return, "return one symbolic relation, not a tuple or list")
    _validate_expression(final_return.value)


def _parameter_names(function: ast.FunctionDef) -> set[str]:
    arguments = function.args
    names = {argument.arg for argument in (*arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs)}
    if arguments.vararg is not None:
        names.add(arguments.vararg.arg)
    if arguments.kwarg is not None:
        names.add(arguments.kwarg.arg)
    return names


def _validate_branch(statement: ast.If, params: set[str], assigned: set[str]) -> str:
    _validate_expression(statement.test)
    if len(statement.body) != 1 or len(statement.orelse) != 1:
        _reject(statement, "each branch must contain exactly one assignment")
    left = _assignment_name(statement.body[0])
    right = _assignment_name(statement.orelse[0])
    if left != right:
        _reject(statement, "both branches must assign the same local variable")
    _validate_expression(_assignment_value(statement.body[0]))
    _validate_expression(_assignment_value(statement.orelse[0]))
    _check_fresh_binding(statement, left, params, assigned)
    return left


def _validate_bindings(function: ast.FunctionDef, params: set[str]) -> set[str]:
    assigned: set[str] = set()
    if_count = 0
    for statement in function.body[:-1]:
        if isinstance(statement, (ast.Assign, ast.AnnAssign)):
            name = _assignment_name(statement)
            _validate_expression(_assignment_value(statement))
            _check_fresh_binding(statement, name, params, assigned)
        elif isinstance(statement, ast.If):
            if_count += 1
            if if_count > 1:
                _reject(statement, "at most one if/else is supported")
            name = _validate_branch(statement, params, assigned)
        else:
            _reject(statement, f"{type(statement).__name__} is not supported")
        assigned.add(name)
    return assigned


def _statement_expressions(statement: ast.stmt) -> list[ast.expr]:
    if isinstance(statement, (ast.Assign, ast.AnnAssign)):
        return [_assignment_value(statement)]
    if isinstance(statement, ast.If):
        return [
            statement.test,
            *(_assignment_value(branch) for branch in (*statement.body, *statement.orelse)),
        ]
    if isinstance(statement, ast.Return) and statement.value is not None:
        return [statement.value]
    return []


def _check_local_reads(function: ast.FunctionDef, params: set[str], assigned: set[str]) -> None:
    """Reject reads of local bindings before their definition."""
    available = set(params)
    local_names = assigned - params
    for statement in function.body:
        for expression in _statement_expressions(statement):
            for subnode in ast.walk(expression):
                if (
                    isinstance(subnode, ast.Name)
                    and isinstance(subnode.ctx, ast.Load)
                    and subnode.id in local_names
                    and subnode.id not in available
                ):
                    _reject(subnode, f"local variable {subnode.id!r} is used before assignment")
        if isinstance(statement, (ast.Assign, ast.AnnAssign)):
            available.add(_assignment_name(statement))
        elif isinstance(statement, ast.If):
            available.add(_assignment_name(statement.body[0]))


def _validate_function(function: ast.FunctionDef) -> None:
    _validate_return(function)
    params = _parameter_names(function)
    assigned = _validate_bindings(function, params)
    _check_local_reads(function, params, assigned)

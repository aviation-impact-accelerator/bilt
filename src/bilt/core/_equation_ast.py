import ast
import copy
import inspect
import linecache
import sys
import textwrap
import typing

from bilt.core._errors import BiltDefinitionError

if typing.TYPE_CHECKING:
    import collections.abc


class BiltEquationSyntaxError(BiltDefinitionError, SyntaxError):
    """Unsupported syntax in an equation."""


def _register_markdown_exec_source(function: collections.abc.Callable[..., object]) -> None:
    """Make Markdown Exec's retained code blocks available to Python's inspector."""
    filename = getattr(getattr(function, "__code__", None), "co_filename", "")
    if not filename.startswith("<code block: "):
        return

    # Markdown Exec retains the source for tracebacks but does not populate
    # linecache, which inspect.getsource() needs.
    formatter = sys.modules.get("markdown_exec._internal.formatters.python")
    code_blocks = getattr(formatter, "_code_blocks", None)
    lines = code_blocks.get(filename) if isinstance(code_blocks, dict) else None
    if not isinstance(lines, list):
        return

    source = "\n".join(lines)
    linecache.cache[filename] = (len(source), None, source.splitlines(keepends=True), filename)


def extract_equation_ast(function: collections.abc.Callable[..., object]) -> ast.FunctionDef:
    """Extract a detached function AST, leaving the original callable unchanged."""
    name = getattr(function, "__name__", type(function).__name__)
    _register_markdown_exec_source(function)
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

    # Avoid reapplying decorators when compiling the tracing function.
    node.decorator_list = []
    return node


def normalise_equation_ast(function: ast.FunctionDef) -> ast.FunctionDef:
    """Validate and rewrite supported if/elif/else and nested conditionals."""
    node = copy.deepcopy(function)
    node.decorator_list = []
    _validate_function(node)

    normalised: list[ast.stmt] = []
    for statement in node.body:
        if not isinstance(statement, ast.If):
            normalised.append(statement)
            continue

        name, value = _normalise_branch(statement)
        # A branch-local annotation is a declaration, not part of the symbolic
        # expression. The merged assignment is deliberately unannotated.
        replacement = ast.Assign(
            targets=[ast.copy_location(ast.Name(id=name, ctx=ast.Store()), statement)],
            value=value,
            type_comment=None,
        )
        normalised.append(ast.copy_location(replacement, statement))

    node.body = normalised
    return ast.fix_missing_locations(node)


def _normalise_branch(statement: ast.stmt) -> tuple[str, ast.expr]:
    if not isinstance(statement, ast.If):
        return _assignment_name(statement), _assignment_value(statement)

    true_name, when_true = _normalise_branch(statement.body[0])
    _, when_false = _normalise_branch(statement.orelse[0])
    expression = ast.Call(
        func=ast.Name(id="_bilt_conditional", ctx=ast.Load()),
        args=[statement.test, when_true, when_false],
        keywords=[],
    )
    return true_name, ast.copy_location(expression, statement)


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


def _branch_statement(statements: list[ast.stmt], conditional: ast.If) -> ast.stmt:
    if len(statements) != 1:
        _reject(conditional, "each branch must contain exactly one assignment or nested if/else")
    return statements[0]


def _branch_binding_name(statement: ast.stmt) -> str:
    if isinstance(statement, ast.If):
        return _branch_binding_name(statement.body[0])
    return _assignment_name(statement)


def _validate_branch_arm(statement: ast.stmt, params: set[str], assigned: set[str]) -> str:
    if isinstance(statement, ast.If):
        return _validate_branch(statement, params, assigned)
    name = _assignment_name(statement)
    _validate_expression(_assignment_value(statement))
    return name


def _validate_branch(statement: ast.If, params: set[str], assigned: set[str]) -> str:
    _validate_expression(statement.test)
    if (
        not isinstance(statement.test, ast.Compare)
        or len(statement.test.ops) != 1
        or not isinstance(statement.test.ops[0], (ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE))
    ):
        _reject(statement.test, "conditional predicate must be one symbolic comparison")

    left = _validate_branch_arm(_branch_statement(statement.body, statement), params, assigned)
    right = _validate_branch_arm(_branch_statement(statement.orelse, statement), params, assigned)
    if left != right:
        _reject(statement, "all branches must assign the same local variable")
    _check_fresh_binding(statement, left, params, assigned)
    return left


def _validate_bindings(function: ast.FunctionDef, params: set[str]) -> set[str]:
    assigned: set[str] = set()
    if_count = 0
    for index, statement in enumerate(function.body[:-1]):
        if (
            index == 0
            and isinstance(statement, ast.Expr)
            and isinstance(statement.value, ast.Constant)
            and isinstance(statement.value.value, str)
        ):
            continue
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
            *(
                expression
                for branch in (*statement.body, *statement.orelse)
                for expression in _statement_expressions(branch)
            ),
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
            available.add(_branch_binding_name(statement.body[0]))


def _validate_function(function: ast.FunctionDef) -> None:
    _validate_return(function)
    params = _parameter_names(function)
    assigned = _validate_bindings(function, params)
    _check_local_reads(function, params, assigned)

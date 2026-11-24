import ast
import copy
import inspect
import linecache
import sys
import textwrap
import typing
lazy import enum

from bilt.core._errors import BiltCategoricalError, BiltDefinitionError
from bilt.core._port import QualifiedPort
from bilt.core._validation import checking_model_definitions

if typing.TYPE_CHECKING:
    import collections.abc


class BiltEquationSyntaxError(BiltDefinitionError, SyntaxError):
    """Unsupported syntax in an equation."""


class BiltCategoricalEquationError(BiltEquationSyntaxError, BiltCategoricalError):
    """An equation uses categorical values outside supported enum matching."""


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


class _EnumMatchContext(typing.NamedTuple):
    types: collections.abc.Mapping[str, type[enum.Enum]]
    namespace: collections.abc.Mapping[str, object]

    def port_type(self, node: ast.AST) -> type[enum.Enum] | None:
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            return self.types.get(node.id)
        if isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load) and isinstance(node.value, ast.Name):
            return self.types.get(QualifiedPort(node.value.id, node.attr).dotted)
        return None

    def is_enum_member(self, node: ast.AST) -> bool:
        if isinstance(node, ast.Name):
            return isinstance(self.namespace.get(node.id), enum.Enum)
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            enum_class = self.namespace.get(node.value.id)
            return (
                isinstance(enum_class, type)
                and issubclass(enum_class, enum.Enum)
                and node.attr in enum_class.__members__
            )
        return False


def normalise_equation_ast(
    function: ast.FunctionDef,
    *,
    enum_types: collections.abc.Mapping[str, type[enum.Enum]] | None = None,
    enum_namespace: collections.abc.Mapping[str, object] | None = None,
) -> ast.FunctionDef:
    """Validate and rewrite supported if/elif/else and match/case conditions."""
    node = copy.deepcopy(function)
    node.decorator_list = []
    params = _parameter_names(node)
    nodes = tuple(item for statement in node.body for item in ast.walk(statement))
    assigned = {item.id for item in nodes if isinstance(item, ast.Name) and isinstance(item.ctx, ast.Store)}
    globals_ = enum_namespace or {}
    namespace = {
        item.id: globals_[item.id]
        for item in nodes
        if isinstance(item, ast.Name) and item.id not in params and item.id not in assigned and item.id in globals_
    }
    enum_context = _EnumMatchContext(enum_types or {}, namespace)
    _validate_function(node, params, nodes, enum_context)

    normalised: list[ast.stmt] = []
    for statement in node.body:
        if not isinstance(statement, (ast.If, ast.Match)):
            normalised.append(statement)
            continue

        name, value = _normalise_branch(statement, {}, enum_context)
        replacement = ast.Assign(
            targets=[ast.copy_location(ast.Name(id=name, ctx=ast.Store()), statement)],
            value=value,
            type_comment=None,
        )
        normalised.append(ast.copy_location(replacement, statement))

    node.body = normalised
    return ast.fix_missing_locations(node)


class _LocalSubstitution(ast.NodeTransformer):
    def __init__(self, bindings: dict[str, ast.expr]) -> None:
        self.bindings = bindings

    def visit_Name(self, node: ast.Name) -> ast.expr:
        if isinstance(node.ctx, ast.Load) and node.id in self.bindings:
            return ast.copy_location(copy.deepcopy(self.bindings[node.id]), node)
        return node


def _substitute_locals(expression: ast.expr, bindings: dict[str, ast.expr]) -> ast.expr:
    return typing.cast("ast.expr", _LocalSubstitution(bindings).visit(copy.deepcopy(expression)))


def _normalise_arm(
    statements: list[ast.stmt], inherited: dict[str, ast.expr], enum_context: _EnumMatchContext
) -> tuple[str, ast.expr]:
    bindings = inherited.copy()
    for statement in statements[:-1]:
        bindings[_assignment_name(statement)] = _substitute_locals(_assignment_value(statement), bindings)

    last = statements[-1]
    if isinstance(last, (ast.If, ast.Match)):
        return _normalise_branch(last, bindings, enum_context)
    return _assignment_name(last), _substitute_locals(_assignment_value(last), bindings)


def _normalise_branch(
    statement: ast.If | ast.Match, bindings: dict[str, ast.expr], enum_context: _EnumMatchContext
) -> tuple[str, ast.expr]:
    if isinstance(statement, ast.Match):
        subject_type = enum_context.port_type(statement.subject)
        name, result = _normalise_arm(statement.cases[-1].body, bindings, enum_context)
        for case in reversed(statement.cases[:-1]):
            _, selected = _normalise_arm(case.body, bindings, enum_context)
            predicate = ast.Compare(
                left=_substitute_locals(statement.subject, bindings),
                ops=[ast.Eq()],
                comparators=[_match_literal(case.pattern, subject_type, enum_context.namespace)],
            )
            result = ast.copy_location(
                ast.Call(
                    func=ast.Name(id="_bilt_conditional", ctx=ast.Load()),
                    args=[ast.copy_location(predicate, case.pattern), selected, result],
                    keywords=[],
                ),
                case.pattern,
            )
        return name, result

    true_name, when_true = _normalise_arm(statement.body, bindings, enum_context)
    _, when_false = _normalise_arm(statement.orelse, bindings, enum_context)
    expression = ast.Call(
        func=ast.Name(id="_bilt_conditional", ctx=ast.Load()),
        args=[_substitute_locals(statement.test, bindings), when_true, when_false],
        keywords=[],
    )
    return true_name, ast.copy_location(expression, statement)


def _reject(
    node: ast.AST, message: str, error_type: type[BiltEquationSyntaxError] = BiltEquationSyntaxError
) -> typing.NoReturn:
    line = getattr(node, "lineno", None)
    location = f" (line {line})" if line is not None else ""
    msg = f"Unsupported equation syntax{location}: {message}"
    raise error_type(msg)


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


def _branch_binding_name(statement: ast.stmt) -> str:
    if isinstance(statement, ast.If):
        return _branch_binding_name(statement.body[-1])
    if isinstance(statement, ast.Match):
        return _branch_binding_name(statement.cases[0].body[-1])
    return _assignment_name(statement)


def _validate_branch_arm(
    statements: list[ast.stmt],
    conditional: ast.If | ast.Match,
    params: set[str],
    assigned: set[str],
    enum_context: _EnumMatchContext,
) -> str:
    if not statements:
        _reject(conditional, "each branch must contain exactly one assignment or nested if/else")

    local_assigned = set(assigned)
    for statement in statements[:-1]:
        if isinstance(statement, (ast.If, ast.Match)):
            _reject(statement, "nested conditional must be last in a branch")
        name = _assignment_name(statement)
        _validate_expression(_assignment_value(statement))
        _check_fresh_binding(statement, name, params, local_assigned)
        local_assigned.add(name)

    last = statements[-1]
    if isinstance(last, (ast.If, ast.Match)):
        return _validate_conditional(last, params, local_assigned, enum_context)
    name = _assignment_name(last)
    _validate_expression(_assignment_value(last))
    _check_fresh_binding(last, name, params, local_assigned)
    return name


def _validate_mask_predicate(node: ast.expr) -> None:
    """Allow comparisons combined with bitwise Boolean mask operators only."""
    if isinstance(node, ast.Compare):
        if len(node.ops) != 1 or not isinstance(node.ops[0], (ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE)):
            _reject(node, "mask predicates require one symbolic comparison")
        _validate_expression(node)
        return
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.BitAnd, ast.BitOr)):
        _validate_mask_predicate(node.left)
        _validate_mask_predicate(node.right)
        return
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Invert):
        _validate_mask_predicate(node.operand)
        return
    _reject(node, "mask predicates require a symbolic comparison, optionally combined with &, | or ~")


def _validate_branch(statement: ast.If, params: set[str], assigned: set[str], enum_context: _EnumMatchContext) -> str:
    _validate_mask_predicate(statement.test)

    left = _validate_branch_arm(statement.body, statement, params, assigned, enum_context)
    right = _validate_branch_arm(statement.orelse, statement, params, assigned, enum_context)
    if left != right:
        _reject(statement, "all branches must assign the same local variable")
    _check_fresh_binding(statement, left, params, assigned)
    return left


def _match_literal(
    pattern: ast.pattern,
    subject_type: type[enum.Enum] | None = None,
    enum_namespace: collections.abc.Mapping[str, object] | None = None,
) -> ast.expr:
    """Resolve numeric or typed enum cases using one validation path.

    Enum members remain qualified values here. Their type and member identity
    are preserved by the symbolic tree, not replaced by an ordinal in the AST.
    """
    if not isinstance(pattern, ast.MatchValue):
        _reject(pattern, "match cases require numeric literals or members of the subject's enum type")
    value = pattern.value
    if subject_type is not None:
        if (
            isinstance(value, ast.Attribute)
            and isinstance(value.value, ast.Name)
            and enum_namespace is not None
            and enum_namespace.get(value.value.id) is subject_type
            and value.attr in subject_type.__members__
        ):
            return copy.deepcopy(value)
        _reject(pattern, "enum cases must name members of the match subject's enum type")

    if isinstance(value, ast.Constant) and type(value.value) in (int, float):
        return copy.deepcopy(value)
    if (
        isinstance(value, ast.UnaryOp)
        and isinstance(value.op, ast.USub)
        and isinstance(value.operand, ast.Constant)
        and type(value.operand.value) in (int, float)
    ):
        return copy.deepcopy(value)
    _reject(pattern, "match cases must be numeric literals followed by a final case _")


def _is_wildcard(pattern: ast.pattern) -> bool:
    return isinstance(pattern, ast.MatchAs) and pattern.pattern is None and pattern.name is None


def _validate_match(statement: ast.Match, params: set[str], assigned: set[str], enum_context: _EnumMatchContext) -> str:
    subject_type = enum_context.port_type(statement.subject)
    if subject_type is None and any(
        enum_context.is_enum_member(item) for case in statement.cases for item in ast.walk(case.pattern)
    ):
        _reject(
            statement.subject,
            "enum patterns require a match subject declared with that enum type",
            BiltCategoricalEquationError,
        )
    try:
        name = _validate_match_arms(statement, params, assigned, enum_context, subject_type)
    except BiltEquationSyntaxError as error:
        if subject_type is None or isinstance(error, BiltCategoricalEquationError):
            raise
        raise BiltCategoricalEquationError(str(error)) from error
    else:
        return name


def _validate_match_arms(
    statement: ast.Match,
    params: set[str],
    assigned: set[str],
    enum_context: _EnumMatchContext,
    subject_type: type[enum.Enum] | None,
) -> str:
    _validate_expression(statement.subject)
    wildcard = _is_wildcard(statement.cases[-1].pattern) if statement.cases else False
    if subject_type is None and (not statement.cases[:-1] or not wildcard):
        _reject(statement, "match requires literal cases and a final case _")
    if subject_type is not None and not statement.cases:
        _reject(statement, "an enum match must have at least one case", BiltCategoricalEquationError)

    names: set[str] = set()
    members: set[str] = set()
    for index, case in enumerate(statement.cases):
        if case.guard is not None:
            _reject(case.guard, "match guards are not supported")
        if not (index == len(statement.cases) - 1 and wildcard):
            _match_literal(case.pattern, subject_type, enum_context.namespace)
            if subject_type is not None:
                # _match_literal verified both enum class and member identity.
                value = typing.cast("ast.MatchValue", case.pattern).value
                members.add(typing.cast("ast.Attribute", value).attr)
        names.add(_validate_branch_arm(case.body, statement, params, assigned, enum_context))
    if subject_type is not None:
        missing = set(subject_type.__members__) - members
        if missing and (not wildcard or checking_model_definitions()):
            msg = (
                f"non-exhaustive {subject_type.__name__} match: "
                f"explicitly handle {', '.join(sorted(missing))}"
            )
            _reject(statement, msg, BiltCategoricalEquationError)
    if len(names) != 1:
        _reject(statement, "all branches must assign the same local variable")
    name = next(iter(names))
    _check_fresh_binding(statement, name, params, assigned)
    return name


def _validate_conditional(
    statement: ast.If | ast.Match, params: set[str], assigned: set[str], enum_context: _EnumMatchContext
) -> str:
    if isinstance(statement, ast.Match):
        return _validate_match(statement, params, assigned, enum_context)
    return _validate_branch(statement, params, assigned, enum_context)


def _validate_bindings(function: ast.FunctionDef, params: set[str], enum_context: _EnumMatchContext) -> set[str]:
    assigned: set[str] = set()
    conditional_count = 0
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
        elif isinstance(statement, (ast.If, ast.Match)):
            conditional_count += 1
            if conditional_count > 1:
                _reject(statement, "at most one if/else or match/case is supported at top level")
            name = _validate_conditional(statement, params, assigned, enum_context)
        else:
            _reject(statement, f"{type(statement).__name__} is not supported")
        assigned.add(name)
    return assigned


def _check_expression_reads(expression: ast.expr, available: set[str], local_names: set[str]) -> None:
    for node in ast.walk(expression):
        if (
            isinstance(node, ast.Name)
            and isinstance(node.ctx, ast.Load)
            and node.id in local_names
            and node.id not in available
        ):
            _reject(node, f"local variable {node.id!r} is used before assignment")


def _check_branch_reads(statement: ast.If | ast.Match, available: set[str], local_names: set[str]) -> None:
    if isinstance(statement, ast.Match):
        _check_expression_reads(statement.subject, available, local_names)
        arms = [case.body for case in statement.cases]
    else:
        _check_expression_reads(statement.test, available, local_names)
        arms = [statement.body, statement.orelse]

    for arm in arms:
        branch_available = set(available)
        for item in arm:
            if isinstance(item, (ast.If, ast.Match)):
                _check_branch_reads(item, branch_available, local_names)
                branch_available.add(_branch_binding_name(item))
            else:
                _check_expression_reads(_assignment_value(item), branch_available, local_names)
                branch_available.add(_assignment_name(item))


def _check_local_reads(function: ast.FunctionDef, params: set[str], assigned: set[str]) -> None:
    """Check each branch independently; only its result name escapes."""
    local_names = set(assigned)
    for node in ast.walk(function):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            local_names.add(_assignment_name(node))

    available = set(params)
    for statement in function.body:
        if isinstance(statement, (ast.Assign, ast.AnnAssign)):
            _check_expression_reads(_assignment_value(statement), available, local_names)
            available.add(_assignment_name(statement))
        elif isinstance(statement, (ast.If, ast.Match)):
            _check_branch_reads(statement, available, local_names)
            available.add(_branch_binding_name(statement))
        elif isinstance(statement, ast.Return) and statement.value is not None:
            _check_expression_reads(statement.value, available, local_names)


def _validate_function(
    function: ast.FunctionDef, params: set[str], nodes: tuple[ast.AST, ...], enum_context: _EnumMatchContext
) -> None:
    _validate_return(function)
    _validate_categorical_expressions(nodes, enum_context)
    assigned = _validate_bindings(function, params, enum_context)
    _check_local_reads(function, params, assigned)


def _validate_categorical_expressions(nodes: tuple[ast.AST, ...], enum_context: _EnumMatchContext) -> None:
    allowed_subjects = {id(statement.subject) for statement in nodes if isinstance(statement, ast.Match)}
    allowed_patterns = {id(item.value) for item in nodes if isinstance(item, ast.MatchValue)}
    for node in nodes:
        if not isinstance(node, (ast.Name, ast.Attribute)) or not isinstance(node.ctx, ast.Load):
            continue
        if id(node) not in allowed_subjects and enum_context.port_type(node) is not None:
            _reject(
                node,
                f"enum parameters may only be used as match subjects. Found {ast.unparse(node)!r}.",
                BiltCategoricalEquationError,
            )
        if id(node) not in allowed_patterns and enum_context.is_enum_member(node):
            _reject(node, "enum members may only be used in match case patterns", BiltCategoricalEquationError)

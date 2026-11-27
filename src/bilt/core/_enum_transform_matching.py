"""Lower scalar enum matches in directed transforms to backend-neutral expressions.

The original Python transform remains untouched. The compiled form compares
encoded ordinals and uses IfExp, which JAX already lowers through lax.cond.
"""

import ast
import copy
import enum
import inspect
import typing

from bilt.core._enum_encoding import enum_code
from bilt.core._errors import BiltCategoricalPortError

if typing.TYPE_CHECKING:
    import collections.abc


def lower_enum_matches(
    function_ast: ast.FunctionDef,
    function: collections.abc.Callable[..., object],
    enum_inputs: collections.abc.Mapping[str, type[enum.Enum]],
    output_type: type[enum.Enum] | None,
) -> ast.FunctionDef:
    """Validate and lower enum matches without changing direct Python execution."""
    closure = inspect.getclosurevars(function)
    bindings = {**closure.globals, **closure.nonlocals}
    lowered = _EnumMatchLowerer(function_ast.name, bindings, enum_inputs, output_type).visit(function_ast)
    return ast.fix_missing_locations(lowered)


class _EnumMatchLowerer(ast.NodeTransformer):
    def __init__(
        self,
        owner: str,
        bindings: dict[str, object],
        enum_inputs: collections.abc.Mapping[str, type[enum.Enum]],
        output_type: type[enum.Enum] | None,
    ) -> None:
        self.owner = owner
        self.bindings = bindings
        self.enum_inputs = enum_inputs
        self.output_type = output_type

    def _reject(self, port: str, detail: str) -> typing.NoReturn:
        raise BiltCategoricalPortError(self.owner, port, detail)

    def _member(self, pattern: ast.pattern, subject_type: type[enum.Enum], port: str) -> enum.Enum:
        if not (
            isinstance(pattern, ast.MatchValue)
            and isinstance(pattern.value, ast.Attribute)
            and isinstance(pattern.value.value, ast.Name)
        ):
            self._reject(port, "match cases must name members of the subject's enum class.")
        owner = self.bindings.get(pattern.value.value.id)
        if owner is not subject_type or pattern.value.attr not in subject_type.__members__:
            self._reject(port, "enum match cases must use members of the exact selector enum class.")
        return subject_type.__members__[pattern.value.attr]

    @staticmethod
    def _wildcard(pattern: ast.pattern) -> bool:
        return isinstance(pattern, ast.MatchAs) and pattern.pattern is None and pattern.name is None

    def _assert_never(self, arm: list[ast.stmt], subject: str) -> bool:
        if len(arm) != 1 or not isinstance(arm[0], ast.Expr) or not isinstance(arm[0].value, ast.Call):
            return False
        call = arm[0].value
        if call.keywords or len(call.args) != 1 or not isinstance(call.args[0], ast.Name):
            return False
        if call.args[0].id != subject:
            return False
        if isinstance(call.func, ast.Name):
            target = self.bindings.get(call.func.id)
        elif isinstance(call.func, ast.Attribute) and isinstance(call.func.value, ast.Name):
            module = self.bindings.get(call.func.value.id)
            target = getattr(module, call.func.attr, None)
        else:
            return False
        return target is typing.assert_never

    def _outcome(self, statements: list[ast.stmt], port: str) -> tuple[str, str | None, ast.expr]:
        if len(statements) != 1:
            self._reject(port, "enum match arms must each contain one return or one assignment.")
        statement = statements[0]
        if isinstance(statement, ast.Return) and statement.value is not None:
            return "return", None, statement.value
        if (
            isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Name)
        ):
            return "assign", statement.targets[0].id, statement.value
        raise BiltCategoricalPortError(
            self.owner, port, "enum match arms must each contain one return or one assignment."
        )

    def _check_output(self, expression: ast.expr, port: str) -> None:
        """Reject statically known wrong categorical results before ordinal conversion."""
        expected = self.output_type
        if expected is None:
            return
        value: object | None = None
        if isinstance(expression, ast.Name):
            value = self.bindings.get(expression.id)
            actual_input = self.enum_inputs.get(expression.id)
            if actual_input is not None and actual_input is not expected:
                self._reject(port, "returns a different enum class than its declared output.")
        elif isinstance(expression, ast.Attribute) and isinstance(expression.value, ast.Name):
            owner = self.bindings.get(expression.value.id)
            if isinstance(owner, type) and issubclass(owner, enum.Enum):
                value = owner.__members__.get(expression.attr)
        if isinstance(value, enum.Enum) and type(value) is not expected:
            self._reject(port, "returns a different enum class than its declared output.")

    def visit_Match(self, node: ast.Match) -> ast.stmt:  # noqa: C901, PLR0912, PLR0915
        if not isinstance(node.subject, ast.Name):
            self._reject("<match>", "enum transforms require a directly named selector.")
        port = node.subject.id
        enum_type = self.enum_inputs.get(port)
        if enum_type is None:
            # Numeric or other Python matches retain their previous behaviour.
            # But an enum pattern with an undeclared subject must not be encoded by accident.
            for case in node.cases:
                value = case.pattern.value if isinstance(case.pattern, ast.MatchValue) else None
                if isinstance(value, ast.Attribute) and isinstance(value.value, ast.Name):
                    owner = self.bindings.get(value.value.id)
                    if isinstance(owner, type) and issubclass(owner, enum.Enum):
                        self._reject(port, "enum match subjects must have a concrete enum annotation.")
            return node
        if not node.cases:
            self._reject(port, "enum match requires one or more enum cases.")
        cases: list[tuple[enum.Enum, ast.expr, str, str | None]] = []
        seen: set[enum.Enum] = set()
        default: tuple[str, str | None, ast.expr] | None = None
        never = False
        for index, case in enumerate(node.cases):
            if case.guard is not None:
                self._reject(port, "enum match guards are not supported.")
            if self._wildcard(case.pattern):
                if index != len(node.cases) - 1:
                    self._reject(port, "the enum wildcard must be the final case.")
                if self._assert_never(case.body, port):
                    never = True
                    continue
                default = self._outcome(case.body, port)
                self._check_output(default[2], port)
                continue
            member = self._member(case.pattern, enum_type, port)
            if member in seen:
                self._reject(port, "duplicate enum match case.")
            seen.add(member)
            kind, target, expression = self._outcome(case.body, port)
            self._check_output(expression, port)
            cases.append((member, expression, kind, target))
        if not cases:
            self._reject(port, "enum match requires at least one named member case.")
        exhaustive = seen == set(enum_type)
        if never and not exhaustive:
            self._reject(port, "assert_never requires cases covering every enum member.")
        if default is None and not exhaustive:
            self._reject(port, "non-exhaustive enum matches require a final case _ with a value.")
        kind, target = cases[0][2:]
        if any((case_kind, case_target) != (kind, target) for _, _, case_kind, case_target in cases):
            self._reject(port, "all enum match arms must return or assign to the same local.")
        if default is not None and (default[0], default[1]) != (kind, target):
            self._reject(port, "the wildcard must return or assign the same local as other cases.")
        result = copy.deepcopy(default[2] if default is not None else cases[-1][1])
        # Since the declared Enum input is checked at the boundary, an
        # exhaustive match can use the final case as an unreachable fallback.
        for member, expression, _, _ in reversed(cases):
            test = ast.Compare(
                left=copy.deepcopy(node.subject),
                ops=[ast.Eq()],
                comparators=[ast.Constant(enum_code(member))],
            )
            conditional = ast.IfExp(test=test, body=copy.deepcopy(expression), orelse=result)
            # Preserve the typed selector for domain-sensitive backend lowering.
            conditional._bilt_enum_selector = port  # pyrefly: ignore[missing-attribute]  # ty: ignore[missing-attribute]  # noqa: SLF001
            result = conditional
        if kind == "return":
            replacement: ast.stmt = ast.Return(value=result)
        else:
            replacement = ast.Assign(targets=[ast.Name(id=typing.cast("str", target), ctx=ast.Store())], value=result)
        return ast.copy_location(replacement, node)

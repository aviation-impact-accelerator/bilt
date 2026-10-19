"""Replace the units written in a transform's body with their coherent values, so a backend computes on bare numbers."""

__all__ = ["erase_units"]

import ast
import inspect
import types
import typing
from collections import Counter

from bilt.core._errors import BiltDefinitionError
from bilt.units import Dimension, Metadata, Quantity, Unit
from bilt.units._category import UnitCategory
from bilt.units._conversion import conversion_between
from bilt.units._errors import BiltUnitError
from bilt.units._units import BASE_UNIT_SYSTEM

if typing.TYPE_CHECKING:
    from collections.abc import Iterator

    from bilt.core._port_declaration import PortDeclaration
    from bilt.core._transform import BasicTransform
    from bilt.units._conversion import Conversion

_UNIT_TYPES = (Quantity, Unit, Dimension)

_QUANTITY_ATTRIBUTES = frozenset({"value", "unit", "convert_to", "cast_to", "strip_metadata"})

_SCOPES = (
    ast.FunctionDef,
    ast.AsyncFunctionDef,
    ast.ClassDef,
    ast.Lambda,
    ast.ListComp,
    ast.SetComp,
    ast.DictComp,
    ast.GeneratorExp,
)


def erase_units(function_ast: ast.FunctionDef, transform: BasicTransform) -> ast.FunctionDef:
    """Replace the units written in a transform's body with their coherent values.

    Raises:
        BiltDefinitionError: If the body holds a construct that cannot be kept faithful without its units.
    """
    shadowed = _shadowed_names(function_ast)
    assigned = {
        node.id
        for node in ast.walk(function_ast)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store) and node not in shadowed
    }
    closure = inspect.getclosurevars(transform._function)  # noqa: SLF001
    parameters = set(transform.__bilt_parameters__) - assigned
    eraser = _UnitEraser(
        bindings={**closure.globals, **closure.nonlocals},
        shadowed=shadowed,
        plain_parameters=_plain_aliases(
            function_ast,
            parameters - transform.__bilt_port_declarations__.keys(),
        ),
        declared_parameters={
            name: declaration
            for name, declaration in transform.__bilt_port_declarations__.items()
            if name in parameters
        },
    )
    for statement in function_ast.body:
        eraser.visit(statement)

    return function_ast


def _plain_aliases(function_ast: ast.FunctionDef, plain: set[str]) -> set[str]:  # noqa: C901
    """Include simple aliases of plain parameters, provided each is bound only once."""
    nodes = [node for node, scope in _scoped(function_ast, function_ast, frozenset()) if scope is function_ast]

    bindings: Counter[str] = Counter()
    for node in nodes:
        match node:
            case ast.Name(id=name, ctx=ast.Store() | ast.Del()) | ast.arg(arg=name):
                bindings[name] += 1

            case ast.FunctionDef(name=name) | ast.AsyncFunctionDef(name=name) | ast.ClassDef(name=name):
                bindings[name] += 1

            case ast.Import(names=names):
                bindings.update(alias.asname or alias.name.split(".")[0] for alias in names)

            case ast.ImportFrom(names=names):
                bindings.update(alias.asname or alias.name for alias in names)

            case (
                ast.ExceptHandler(name=str() as name)
                | ast.MatchAs(name=str() as name)
                | ast.MatchStar(name=str() as name)
                | ast.MatchMapping(rest=str() as name)
            ):
                bindings[name] += 1

    candidates: dict[str, str] = {}
    for node in nodes:
        match node:
            case (
                ast.Assign(
                    targets=[ast.Name(id=target)],
                    value=ast.Name(id=source),
                )
                | ast.AnnAssign(
                    target=ast.Name(id=target),
                    value=ast.Name(id=source),
                )
            ):
                if bindings[target] == 1:
                    candidates[target] = source

    while True:
        expanded = plain | {target for target, source in candidates.items() if source in plain}
        if expanded == plain:
            return plain
        plain = expanded


def _shadowed_names(function_ast: ast.FunctionDef) -> set[ast.Name]:
    """The names in `function_ast` that a scope nested inside it binds, so they do not read its parameters."""
    scope_of = dict(_scoped(function_ast, function_ast, frozenset()))
    bound: dict[ast.AST, set[str]] = {scope: set() for scope in scope_of.values()}
    for node, scope in scope_of.items():
        match node:
            case ast.arg(arg=name) | ast.Name(id=name, ctx=ast.Store()):
                bound[scope].add(name)
    visible: dict[ast.AST, frozenset[str]] = {function_ast: frozenset()}
    for node in scope_of:
        if isinstance(node, _SCOPES) and node is not function_ast:
            visible[node] = visible[scope_of[node]] | bound[node]
    return {node for node, scope in scope_of.items() if isinstance(node, ast.Name) and node.id in visible[scope]}


def _scoped(node: ast.AST, scope: ast.AST, outer: frozenset[ast.AST]) -> Iterator[tuple[ast.AST, ast.AST]]:
    """Each of `node` and its descendants with the scope that evaluates it, given `scope` evaluates `node` itself."""
    if node in outer:
        return
    yield node, scope
    outside = _evaluated_outside(node)
    for expression in outside:
        yield from _scoped(expression, scope, outer)
    inner = node if isinstance(node, _SCOPES) else scope
    nested = outer | frozenset(outside) if outside else outer
    for child in ast.iter_child_nodes(node):
        yield from _scoped(child, inner, nested)


def _evaluated_outside(scope: ast.AST) -> list[ast.expr]:
    """The sub-expressions of `scope` that its enclosing scope evaluates, so `scope` does not bind a name there."""
    match scope:
        case ast.ListComp() | ast.SetComp() | ast.GeneratorExp() | ast.DictComp():
            return [scope.generators[0].iter]
        case ast.Lambda(args=arguments):
            return _signature(arguments)
        case ast.FunctionDef(args=arguments, returns=returns) | ast.AsyncFunctionDef(args=arguments, returns=returns):
            return [*scope.decorator_list, *_signature(arguments), *([returns] if returns else [])]
        case ast.ClassDef():
            return [*scope.decorator_list, *scope.bases, *(keyword.value for keyword in scope.keywords)]
    return []


def _signature(arguments: ast.arguments) -> list[ast.expr]:
    """The defaults and annotations of `arguments`, which the scope enclosing their function resolves."""
    return [
        *arguments.defaults,
        *(default for default in arguments.kw_defaults if default is not None),
        *(node.annotation for node in ast.walk(arguments) if isinstance(node, ast.arg) and node.annotation),
    ]


class _UnitEraser(ast.NodeTransformer):
    def __init__(
        self,
        bindings: dict[str, object],
        shadowed: set[ast.Name],
        plain_parameters: set[str],
        declared_parameters: dict[str, PortDeclaration],
    ) -> None:
        self.bindings = bindings
        self.shadowed = shadowed
        self.plain_parameters = plain_parameters
        self.declared_parameters = declared_parameters

    def visit(self, node: ast.AST) -> ast.AST:
        return self._erased(node) if isinstance(node, ast.expr) else self.generic_visit(node)

    def _erased(self, node: ast.expr) -> ast.expr:  # noqa: C901, PLR0911
        if self._is_left_to_python(node):
            if isinstance(node, (ast.Attribute, ast.Call)):
                node.__dict__["_bilt_preserved_names"] = {
                    child.id
                    for child in ast.walk(node)
                    if isinstance(child, ast.Name)
                    and isinstance(child.ctx, ast.Load)
                    and child not in self.shadowed
                    and child.id in self.bindings
                }
            return node

        if self._is_constant(node):
            return self._literal(node, self._evaluated(node)) if self._reaches_a_unit(node) else node

        match node:
            case ast.Attribute(
                value=ast.Call(func=ast.Attribute(attr="convert_to"), args=[unit], keywords=[]) as converted,
                attr="value",
            ):
                conversion = self._conversion(unit, node)
                coherent = self._erased(converted)
                if conversion.offset:
                    coherent = ast.BinOp(left=coherent, op=ast.Sub(), right=ast.Constant(conversion.offset))
                return ast.BinOp(left=coherent, op=ast.Div(), right=ast.Constant(conversion.scale))
            case ast.Call(func=ast.Attribute(value=quantity, attr="convert_to"), args=[unit], keywords=[]):
                target = self._unit(unit, node)
                declaration = (
                    self.declared_parameters.get(quantity.id)
                    if isinstance(quantity, ast.Name) and quantity not in self.shadowed
                    else None
                )
                if declaration is not None and (
                    target.dimension != declaration.dimensionality
                    or not (target.category is UnitCategory.ABSOLUTE if declaration.absolute else target.is_relative)
                ):
                    msg = (
                        f"`{ast.unparse(node)}` contradicts the signature: `{ast.unparse(quantity)}` is a Quantity "
                        f"of {declaration}, which converts only to "
                        f"{'an absolute' if declaration.absolute else 'a relative'} unit such as "
                        f"{declaration.coherent_unit}"
                    )
                    raise BiltDefinitionError(msg)
                return self._erased(quantity)
            case (
                ast.Call(func=function, args=[value, unit], keywords=[])
                | ast.Call(func=function, args=[value], keywords=[ast.keyword(arg="unit", value=unit)])
                | ast.Call(
                    func=function,
                    args=[],
                    keywords=[ast.keyword(arg="value", value=value), ast.keyword(arg="unit", value=unit)]
                    | [ast.keyword(arg="unit", value=unit), ast.keyword(arg="value", value=value)],
                )
            ) if self._resolved(function) is Quantity:
                conversion = self._conversion(unit, node)
                coherent = ast.BinOp(left=self._erased(value), op=ast.Mult(), right=ast.Constant(conversion.scale))
                if conversion.offset:
                    coherent = ast.BinOp(left=coherent, op=ast.Add(), right=ast.Constant(conversion.offset))
                return coherent
            case ast.Call(func=ast.Attribute(value=quantity, attr="strip_metadata"), args=[], keywords=[]):
                return self._erased(quantity)
            case ast.BinOp(left=quantity, op=ast.Mult(), right=tag) if isinstance(self._resolved(tag), Metadata):
                return self._erased(quantity)
            case ast.Attribute(value=quantity, attr=attribute) if attribute in _QUANTITY_ATTRIBUTES:
                msg = f"`{ast.unparse(node)}` {_attribute_problem(quantity, attribute)}"
                raise BiltDefinitionError(msg)

        return typing.cast("ast.expr", self.generic_visit(node))

    def _is_constant(self, node: ast.expr) -> bool:  # noqa: PLR0911
        """Whether this is a supported constant expression."""
        match node:
            case ast.Constant():
                return True

            case ast.Name(id=name):
                if node in self.shadowed or name not in self.bindings:
                    return False

                value = self.bindings[name]
                return type(value) in (bool, int, float) or isinstance(value, (Quantity, Unit, Dimension, Metadata))

            case ast.Attribute(value=root):
                # Resolve named constants through imported modules.
                if isinstance(self._resolved(root), types.ModuleType):
                    value = self._resolved(node)
                    return type(value) in (bool, int, float) or isinstance(value, (Quantity, Unit, Dimension, Metadata))

                # Supported, read-only properties on known Bilt constants.
                return isinstance(self._resolved(root), (Quantity, Unit, Dimension)) and node.attr in {
                    "value",
                    "dimension",
                    "is_dimensionless",
                    "is_relative",
                    "category",
                }

            case ast.UnaryOp(op=ast.UAdd() | ast.USub(), operand=operand):
                return self._is_constant(operand)

            case ast.BinOp(
                left=left,
                op=ast.Add() | ast.Sub() | ast.Mult() | ast.Div() | ast.Pow(),
                right=right,
            ):
                return self._is_constant(left) and self._is_constant(right)

            case ast.Call(func=function, args=args, keywords=keywords):
                return (
                    self._resolved(function) is Quantity
                    and all(self._is_constant(arg) for arg in args)
                    and all(
                        keyword.arg in {"value", "unit"} and self._is_constant(keyword.value) for keyword in keywords
                    )
                )

            case _:
                return False

    def _reaches_a_unit(self, node: ast.expr) -> bool:
        """Whether `node` names a quantity, a unit or a dimensionality, or one of their classes."""
        return any(
            isinstance(value, _UNIT_TYPES) or any(value is unit_type for unit_type in _UNIT_TYPES)
            for value in map(self._resolved, ast.walk(node))
        )

    def _resolved(self, node: ast.AST) -> object:
        match node:
            case ast.Name(id=name) if node not in self.shadowed:
                return self.bindings.get(name)
            case ast.Attribute(value=root, attr=attribute) if isinstance(
                module := self._resolved(root), types.ModuleType
            ):
                return getattr(module, attribute, None)
        return None

    def _is_left_to_python(self, node: ast.expr) -> bool:
        """Whether `node` reads a quantity attribute of a plain parameter, calls one, reads `.value` off such a call, or multiplies one by a tag."""
        match node:
            case (
                ast.Attribute(value=ast.Name() as name, attr=attribute)
                | ast.Call(func=ast.Attribute(value=ast.Name() as name, attr=attribute))
                | ast.Attribute(
                    value=ast.Call(func=ast.Attribute(value=ast.Name() as name, attr=attribute)), attr="value"
                )
            ):
                return (
                    name.id in self.plain_parameters and name not in self.shadowed and attribute in _QUANTITY_ATTRIBUTES
                )
            case ast.BinOp(left=ast.Name() as name, op=ast.Mult(), right=tag):
                return (
                    name.id in self.plain_parameters
                    and name not in self.shadowed
                    and isinstance(self._resolved(tag), Metadata)
                )
        return False

    def _evaluated(self, node: ast.expr) -> object:
        return eval(compile(ast.Expression(body=node), "<transform>", "eval"), self.bindings)  # noqa: S307

    def _literal(self, node: ast.expr, value: object) -> ast.expr:
        match value:
            case Unit(category=UnitCategory.ABSOLUTE):
                problem = (
                    "is an absolute unit, which cannot be used as a scale: "
                    f"write `Quantity(<value>, {ast.unparse(node)})`"
                )
            case Unit():
                return self._literal(node, Quantity(1.0, value))
            case Quantity():
                return ast.Constant(BASE_UNIT_SYSTEM.coherent_value(value))
            case bool() | int() | float():
                return ast.Constant(value)
            case _:
                problem = "is not a quantity, a unit or a number, so a transform cannot compute with it"
        msg = f"`{ast.unparse(node)}` {problem}"
        raise BiltDefinitionError(msg)

    def _unit(self, unit: ast.expr, construct: ast.expr) -> Unit:
        """The unit that `construct` names."""
        if self._is_constant(unit) and isinstance(value := self._evaluated(unit), Unit):
            return value
        problem = f"is not a unit named outside the transform, so `{ast.unparse(construct)}` cannot use it"
        msg = f"`{ast.unparse(unit)}` {problem}: name the unit directly"
        raise BiltDefinitionError(msg)

    def _conversion(self, unit: ast.expr, construct: ast.expr) -> Conversion:
        """The conversion to a coherent value from the unit that `construct` names."""
        named = self._unit(unit, construct)
        coherent_unit = BASE_UNIT_SYSTEM.coherent_unit(named.dimension)
        try:
            return conversion_between(named, coherent_unit if named.is_relative else abs(coherent_unit))
        except BiltUnitError as error:
            problem = f"cannot be converted to a coherent unit, so `{ast.unparse(construct)}` cannot use it"
            msg = f"`{ast.unparse(unit)}` {problem}: {error}"
            raise BiltDefinitionError(msg) from error


def _attribute_problem(quantity: ast.expr, attribute: str) -> str:
    conversion = ast.unparse(
        ast.Call(func=ast.Attribute(value=quantity, attr="convert_to"), args=[ast.Name(id="<unit>")])
    )
    shed = ast.unparse(ast.Call(func=ast.Attribute(value=quantity, attr="strip_metadata"), args=[]))
    match attribute:
        case "value":
            return f"depends on the unit `{ast.unparse(quantity)}` arrived in: write `{conversion}.value`"
        case "unit":
            return f"depends on the unit `{ast.unparse(quantity)}` arrived in: name the unit you mean instead"
        case "convert_to":
            return f"must be called with one unit: write `{conversion}`"
        case "cast_to":
            attach = ast.unparse(ast.BinOp(left=quantity, op=ast.Mult(), right=ast.Name(id="<tag>")))
            return (
                f"depends on the unit `{ast.unparse(quantity)}` arrived in: "
                f"shed its tags with `{shed}` or attach one with `{attach}`"
            )
        case _:
            return f"must be called with no arguments: write `{shed}`"

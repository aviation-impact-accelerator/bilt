"""Replace the units written in a transform's body with their coherent values, so a backend computes on bare numbers."""

__all__ = ["erase_units"]

import ast
import inspect
import types
import typing

from bilt.core._errors import BiltDefinitionError
from bilt.units import Dimension, Quantity, Unit
from bilt.units._category import UnitCategory
from bilt.units._conversion import conversion_between
from bilt.units._errors import BiltUnitError
from bilt.units._units import BASE_UNIT_SYSTEM

if typing.TYPE_CHECKING:
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
    assigned = {
        node.id for node in ast.walk(function_ast) if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
    }
    shadowed: set[ast.Name] = set()
    for scope in ast.walk(function_ast):
        if isinstance(scope, _SCOPES) and scope is not function_ast:
            names = [node for node in ast.walk(scope) if isinstance(node, ast.Name)]
            bound = {node.arg for node in ast.walk(scope) if isinstance(node, ast.arg)} | {
                name.id for name in names if isinstance(name.ctx, ast.Store)
            }
            shadowed.update(name for name in names if name.id in bound)
    closure = inspect.getclosurevars(transform._function)  # noqa: SLF001
    parameters = set(transform.__bilt_parameters__) - assigned
    eraser = _UnitEraser(
        bindings={**closure.globals, **closure.nonlocals},
        shadowed=shadowed,
        plain_parameters=parameters - transform.__bilt_port_declarations__.keys(),
        declared_parameters={
            name: declaration
            for name, declaration in transform.__bilt_port_declarations__.items()
            if name in parameters
        },
    )
    for statement in function_ast.body:
        eraser.visit(statement)

    return function_ast


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

    def _erased(self, node: ast.expr) -> ast.expr:
        if self._is_constant(node):
            return self._literal(node, self._evaluated(node)) if self._reaches_a_unit(node) else node

        if not self._is_left_to_python(node):
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
                        or not (
                            target.category is UnitCategory.ABSOLUTE if declaration.absolute else target.is_relative
                        )
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
                case ast.Attribute(value=quantity, attr=attribute) if attribute in _QUANTITY_ATTRIBUTES:
                    msg = f"`{ast.unparse(node)}` {_attribute_problem(quantity, attribute)}"
                    raise BiltDefinitionError(msg)

        return typing.cast("ast.expr", self.generic_visit(node))

    def _is_constant(self, node: ast.expr) -> bool:
        return all(
            name.id in self.bindings and name not in self.shadowed
            for name in ast.walk(node)
            if isinstance(name, ast.Name)
        )

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
        """Whether `node` reads a quantity attribute of a plain parameter, calls one, or reads `.value` off such a call."""
        match node:
            case (
                ast.Attribute(value=ast.Name(id=name), attr=attribute)
                | ast.Call(func=ast.Attribute(value=ast.Name(id=name), attr=attribute))
                | ast.Attribute(
                    value=ast.Call(func=ast.Attribute(value=ast.Name(id=name), attr=attribute)), attr="value"
                )
            ):
                return name in self.plain_parameters and attribute in _QUANTITY_ATTRIBUTES
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
    match attribute:
        case "value":
            return f"depends on the unit `{ast.unparse(quantity)}` arrived in: write `{conversion}.value`"
        case "unit":
            return f"depends on the unit `{ast.unparse(quantity)}` arrived in: name the unit you mean instead"
        case "convert_to":
            return f"must be called with one unit: write `{conversion}`"
        case _:
            return "relabels a quantity, which a transform cannot do: drop the call"

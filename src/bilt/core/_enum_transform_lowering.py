"""Lower enum constants to backend-native scalar ordinals in transforms."""

import ast
import enum
import inspect
import typing

from bilt.core._enum_encoding import enum_code

if typing.TYPE_CHECKING:
    import collections.abc


def lower_enum_literals(function_ast: ast.FunctionDef, function: collections.abc.Callable[..., object]) -> ast.FunctionDef:
    """Compiled transform inputs are encoded; make literal members match that representation."""
    closure = inspect.getclosurevars(function)
    bindings = closure.globals | closure.nonlocals
    local_names = {
        node.id for node in ast.walk(function_ast) if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
    } | {node.arg for node in ast.walk(function_ast) if isinstance(node, ast.arg)}
    return typing.cast("ast.FunctionDef", _EnumLiteralLowerer(bindings, local_names).visit(function_ast))


class _EnumLiteralLowerer(ast.NodeTransformer):
    def __init__(self, bindings: dict[str, object], local_names: set[str]) -> None:
        self.bindings = bindings
        self.local_names = local_names

    def visit_Attribute(self, node: ast.Attribute) -> ast.expr:
        if isinstance(node.value, ast.Name) and node.value.id not in self.local_names:
            enum_type = self.bindings.get(node.value.id)
            if isinstance(enum_type, type) and issubclass(enum_type, enum.Enum) and node.attr in enum_type.__members__:
                return ast.copy_location(ast.Constant(enum_code(enum_type.__members__[node.attr])), node)
        return typing.cast("ast.expr", self.generic_visit(node))

    def visit_Name(self, node: ast.Name) -> ast.expr:
        if isinstance(node.ctx, ast.Load) and node.id not in self.local_names:
            member = self.bindings.get(node.id)
            if isinstance(member, enum.Enum):
                return ast.copy_location(ast.Constant(enum_code(member)), node)
        return node

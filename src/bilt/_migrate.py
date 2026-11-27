"""Migration rules for the transition to the quantity protocol API."""

import ast
import typing

from bilt._migration import migrate as _migrate
from bilt._migration import migrate_source as _migrate_source

if typing.TYPE_CHECKING:
    from pathlib import Path

    from bilt._migration import Migration, MigrationContext, MigrationRule

_QUANTITY = frozenset({"bilt.Quantity", "bilt.units.Quantity"})
_TENSOR = "bilt.QuantityTensor"
_PAIR = 2
_DIMENSIONS = frozenset(
    {
        "AMOUNT",
        "AREA",
        "CHARGE",
        "CONDUCTANCE",
        "CURRENCY",
        "CURRENT",
        "DIMENSIONLESS",
        "DYNAMIC_VISCOSITY",
        "ENERGY",
        "FORCE",
        "LENGTH",
        "LUMINOSITY",
        "MASS",
        "POWER",
        "PRESSURE",
        "RESISTANCE",
        "TEMPERATURE",
        "TIME",
        "VOLTAGE",
        "VOLUME",
    }
)



def _arguments(node: ast.Call, parameters: tuple[str, ...]) -> dict[str, ast.expr] | None:
    if len(node.args) > len(parameters) or any(isinstance(arg, ast.Starred) for arg in node.args):
        return None
    result = dict(zip(parameters, node.args, strict=False))
    for keyword in node.keywords:
        if keyword.arg is None or keyword.arg not in parameters or keyword.arg in result:
            return None
        result[keyword.arg] = keyword.value
    return result


class QuantityConstructorRule:
    """Rewrite legacy scalar quantity construction as multiplication by a unit."""

    def apply(self, node: ast.expr, context: MigrationContext) -> None:
        if not isinstance(node, ast.Call) or context.resolve(node.func) not in _QUANTITY:
            return
        args = _arguments(node, ("value", "unit"))
        # Keep evaluation order: unit= before value= may have side effects.
        if args is not None and tuple(args) == ("value", "unit"):
            value, unit = args.values()
            replacement = b"((" + context.render(value) + b") * (" + context.render(unit) + b"))"
            if context.replace(node, replacement, multiline=True):
                return
        context.note(node, f"review obsolete {context.resolve(node.func)} constructor manually")


class TensorConstructorRule:
    """Rewrite legacy tensor construction using the public tensor factory."""

    def apply(self, node: ast.expr, context: MigrationContext) -> None:
        if not isinstance(node, ast.Call):
            return
        kind = context.resolve(node.func)
        if kind not in {"bilt.Tensor", "bilt.QuantityTensor", "bilt.Tensor.scalar"}:
            return
        if not self._rewrite(node, context, scalar=kind == "bilt.Tensor.scalar", dimensioned=kind == _TENSOR):
            context.note(node, f"review obsolete {kind} constructor manually")

    @staticmethod
    def _rewrite(
        node: ast.Call, context: MigrationContext, *, scalar: bool, dimensioned: bool
    ) -> bool:
        parameters = ("value", "unit") if scalar else ("values", "domain", "unit")
        args = _arguments(node, parameters)
        if args is None or parameters[0] not in args or (not scalar and "domain" not in args):
            return False
        if dimensioned and "unit" not in args:
            return False
        if b"#" in context.text(node):
            return False
        factory = context.ensure_import("bilt", "tensor")
        parts = [
            (("values" if name == "value" else name).encode() + b"=" + context.render(arg))
            for name, arg in args.items()
        ]
        if scalar and next(iter(args)) == "value":
            parts[0] = context.render(args["value"])
        return context.replace(node, factory + b"(" + b", ".join(parts) + b")", multiline=True)


class QuantityUnionRule:
    """Simplify the old scalar-or-tensor quantity union."""

    def apply(self, node: ast.expr, context: MigrationContext) -> None:
        if not isinstance(node, ast.BinOp) or not isinstance(node.op, ast.BitOr):
            return
        left = context.resolve(node.left)
        right = context.resolve(node.right)
        if left in _QUANTITY and right == _TENSOR:
            context.replace(node, context.text(node.left), multiline=True)
        elif right in _QUANTITY and left == _TENSOR:
            context.replace(node, context.text(node.right), multiline=True)


class DimensionedTensorAnnotationRule:
    """Use the dimensioned tensor protocol for explicit physical dimensions."""

    def apply(self, node: ast.expr, context: MigrationContext) -> None:
        if (
            not isinstance(node, ast.Subscript)
            or context.resolve(node.value) != "typing.Annotated"
            or not isinstance(node.slice, ast.Tuple)
            or not node.slice.elts
        ):
            return
        base, *metadata = node.slice.elts
        if context.resolve(base) != "bilt.Tensor":
            return
        for item in metadata:
            path = context.resolve(item)
            if path is not None and path.startswith("bilt.units.dimensions.") and path.rsplit(".", 1)[-1] in _DIMENSIONS:
                context.replace(base, context.ensure_import("bilt", "QuantityTensor"))
                return


class TensorRuntimeCheckRule:
    """Preserve legacy runtime checks for explicitly mixed tensor parameters."""

    def apply(self, node: ast.expr, context: MigrationContext) -> None:
        if (
            not isinstance(node, ast.Call)
            or not isinstance(node.func, ast.Name)
            or node.func.id != "isinstance"
            or not context.builtin_visible("isinstance")
            or node.keywords
            or len(node.args) != _PAIR
            or context.resolve(node.args[1]) != "bilt.Tensor"
        ):
            return
        value, target = node.args
        members = context.parameter_members.get(value.id, set()) if isinstance(value, ast.Name) else set()
        if "bilt.Tensor" in members and members & (_QUANTITY | {_TENSOR}):
            quantity_tensor = context.ensure_import("bilt", "QuantityTensor")
            if context.replace(target, b"(" + context.text(target) + b", " + quantity_tensor + b")"):
                return
        context.note(node, "review Tensor runtime check for unit-bearing values")


QUANTITY_RULES: tuple[MigrationRule, ...] = (
    QuantityConstructorRule(),
    TensorConstructorRule(),
    QuantityUnionRule(),
    DimensionedTensorAnnotationRule(),
    TensorRuntimeCheckRule(),
)


def migrate_source(source: str) -> Migration:
    """Apply the quantity protocol migration to a Python source string."""
    return _migrate_source(source, QUANTITY_RULES)


def migrate(target: Path, *, write: bool) -> int:
    """Preview or apply the quantity protocol migration."""
    return _migrate(target, QUANTITY_RULES, write=write)

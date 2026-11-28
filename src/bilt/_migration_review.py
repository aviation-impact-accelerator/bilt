"""Review shape-sensitive boundaries without guessing the intended model behaviour."""

import ast
from bilt._migration import MigrationContext

_QUANTITY = frozenset({"bilt.Quantity", "bilt.units.Quantity"})
_TENSOR = "bilt.QuantityTensor"
_PAIR = 2


def _runtime_check(node: ast.expr, context: MigrationContext) -> str | None:
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "isinstance"
        and context.builtin_visible("isinstance")
        and len(node.args) == _PAIR
        and not node.keywords
    ):
        return context.resolve(node.args[1])
    return None


def _scalar_unit_product(node: ast.expr, context: MigrationContext) -> bool:
    if not isinstance(node, ast.BinOp) or not isinstance(node.op, ast.Mult):
        return False
    if not isinstance(node.left, ast.Constant) or type(node.left.value) not in {int, float}:
        return False
    unit = context.resolve(node.right)
    return unit is not None and unit.startswith("bilt.units.")


class QuantityShapeReview:
    """Identify changed runtime checks and known scalar-construction boundaries."""

    def __init__(self, tree: ast.Module) -> None:
        self.parents = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}

    def apply(self, node: ast.expr, context: MigrationContext) -> None:
        kind = _runtime_check(node, context)
        if kind in _QUANTITY:
            context.note(node, "review Quantity runtime check; it now also accepts quantity tensors")
        elif kind == _TENSOR:
            parent = self.parents.get(node)
            if isinstance(parent, ast.UnaryOp) and isinstance(parent.op, ast.Not):
                context.note(node, "review scalar assumptions; excluding QuantityTensor does not narrow Quantity.value")
        elif isinstance(node, ast.Attribute) and node.attr == "value":
            self._review_magnitude(node, context)

    def _review_magnitude(self, node: ast.Attribute, context: MigrationContext) -> None:
        receiver = node.value
        if isinstance(receiver, ast.Call) and isinstance(receiver.func, ast.Attribute):
            if receiver.func.attr == "convert_to":
                receiver = receiver.func.value
        if not _scalar_unit_product(receiver, context):
            return
        parent = self.parents.get(node)
        if (
            isinstance(parent, ast.Call)
            and isinstance(parent.func, ast.Name)
            and parent.func.id == "float"
            and context.builtin_visible("float")
        ):
            return
        context.note(node, "review scalar magnitude use; Quantity.value is Number, use float(...) if scalar is intended")


def review_quantity_shapes(source: str) -> tuple[str, ...]:
    """Report syntactically identifiable shape-sensitive sites in migrated source."""
    tree = ast.parse(source)
    visitor = MigrationContext(source, tree, (QuantityShapeReview(tree),))
    visitor.visit(tree)
    return tuple(visitor.notes)

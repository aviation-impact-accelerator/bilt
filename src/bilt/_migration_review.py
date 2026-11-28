"""Review shape-sensitive boundaries without guessing the intended model behaviour."""

import ast
from typing import TYPE_CHECKING

from bilt._migration import MigrationContext

if TYPE_CHECKING:
    from collections.abc import Iterator

_QUANTITY = frozenset({"bilt.Quantity", "bilt.units.Quantity"})
_TENSOR = "bilt.QuantityTensor"
_PAIR = 2
_NUMPY_BOUNDARIES = frozenset({"numpy.broadcast_to", "numpy.testing.assert_allclose"})


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


def _quantity_receiver(node: ast.Attribute) -> ast.expr:
    receiver = node.value
    if (
        isinstance(receiver, ast.Call)
        and isinstance(receiver.func, ast.Attribute)
        and receiver.func.attr in {"convert_to", "cast_to"}
    ):
        return receiver.func.value
    return receiver


def _checks_receiver(
    test: ast.expr, receiver: ast.expr, context: MigrationContext, kinds: frozenset[str]
) -> bool:
    return (
        isinstance(test, ast.Call)
        and _runtime_check(test, context) in kinds
        and ast.dump(test.args[0]) == ast.dump(receiver)
    )


class QuantityShapeReview:
    """Identify magnitude uses that may require an explicit shape decision."""

    def __init__(self, tree: ast.Module) -> None:
        self.parents = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}

    def _ancestors(self, node: ast.AST) -> Iterator[tuple[ast.AST, ast.AST]]:
        while (parent := self.parents.get(node)) is not None:
            yield node, parent
            if isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
                return
            node = parent

    def _has_quantity_evidence(self, node: ast.Attribute, receiver: ast.expr, context: MigrationContext) -> bool:
        if isinstance(receiver, ast.Name) and context.parameter_members.get(receiver.id, set()) & _QUANTITY:
            return True
        for child, parent in self._ancestors(node):
            if (
                isinstance(parent, ast.If)
                and child in parent.body
                and isinstance(parent.test, ast.UnaryOp)
                and isinstance(parent.test.op, ast.Not)
                and _checks_receiver(parent.test.operand, receiver, context, frozenset({_TENSOR}))
            ):
                return True
            if not isinstance(child, ast.stmt):
                continue
            for _, suite in ast.iter_fields(parent):
                if not isinstance(suite, list) or child not in suite:
                    continue
                for statement in suite[: suite.index(child)]:
                    if isinstance(statement, ast.Assert) and _checks_receiver(
                        statement.test, receiver, context, _QUANTITY
                    ):
                        return True
        return False

    def _scalar_return(self, node: ast.Attribute, context: MigrationContext) -> bool:
        if not isinstance(self.parents.get(node), ast.Return) or not context.builtin_visible("float"):
            return False
        for _, parent in self._ancestors(node):
            if isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef)):
                annotation = parent.returns
                return (isinstance(annotation, ast.Name) and annotation.id == "float") or (
                    isinstance(annotation, ast.Constant) and annotation.value == "float"
                )
        return False

    def _numpy_boundary(self, node: ast.Attribute, context: MigrationContext) -> bool:
        parent = self.parents.get(node)
        if isinstance(parent, ast.keyword):
            parent = self.parents.get(parent)
        return isinstance(parent, ast.Call) and context.resolve(parent.func) in _NUMPY_BOUNDARIES

    def apply(self, node: ast.expr, context: MigrationContext) -> None:
        if not isinstance(node, ast.Attribute) or node.attr != "value":
            return
        receiver = _quantity_receiver(node)
        parent = self.parents.get(node)
        if (
            isinstance(parent, ast.Call)
            and isinstance(parent.func, ast.Name)
            and parent.func.id == "float"
            and context.builtin_visible("float")
        ):
            return
        if _scalar_unit_product(receiver, context):
            context.note(
                node, "review scalar magnitude use; Quantity.value is Number, use float(...) if scalar is intended"
            )
        elif self._has_quantity_evidence(node, receiver, context) and (
            self._scalar_return(node, context) or self._numpy_boundary(node, context)
        ):
            context.note(
                node, "review magnitude at scalar or NumPy boundary; explicitly narrow or extract the intended shape"
            )


def review_quantity_shapes(source: str) -> tuple[str, ...]:
    """Report syntactically identifiable shape-sensitive sites in migrated source."""
    tree = ast.parse(source)
    visitor = MigrationContext(source, tree, (QuantityShapeReview(tree),))
    visitor.visit(tree)
    return tuple(visitor.notes)

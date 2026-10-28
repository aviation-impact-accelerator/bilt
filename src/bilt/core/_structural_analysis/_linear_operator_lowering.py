"""Split lowered linear tensor expressions while retaining their broadcast shapes."""

import ast
import dataclasses
import typing

from bilt._backend._linear_solve import LINEAR_EXPAND_INTRINSIC_NAME, LINEAR_ZERO_INTRINSIC_NAME
from bilt.core._port import mangle_name
from bilt.core._structural_analysis._linear_block import BiltLinearBlockError
from bilt.core._to_ast import tree_to_functiondef
from bilt.symbolic._atoms import ADD, MUL, NEG, IndexOpType, ReductionOpType, SymbolType
from bilt.symbolic._tree import Tree, symbol_tree


@dataclasses.dataclass(frozen=True)
class OperatorParts:
    """A linear action on the identity basis and a shape-preserving constant part."""

    linear: ast.expr
    constant: ast.expr


def expression_ast(tree: Tree) -> ast.expr:
    function = tree_to_functiondef(tree, name="__bilt_operator__", parameter_order=())
    return typing.cast("ast.expr", typing.cast("ast.Return", function.body[0]).value)


@dataclasses.dataclass
class OperatorLowering:
    unknowns: tuple[str, ...]
    parameters: tuple[str, ...]
    basis_name: str
    zero_name: str
    _dependencies: dict[Tree, bool] = dataclasses.field(default_factory=dict)

    def zeros(self) -> ast.Assign:
        """Give each constant-part unknown a zero with its canonical tensor shape."""
        values: list[ast.expr] = [
            _zero(_item(self.basis_name, index), basis=True) for index in range(len(self.unknowns))
        ]
        return ast.Assign(
            targets=[ast.Name(id=self.zero_name, ctx=ast.Store())], value=ast.Tuple(elts=values, ctx=ast.Load())
        )

    def depends(self, tree: Tree) -> bool:
        if tree not in self._dependencies:
            self._dependencies[tree] = (
                tree.is_leaf and tree.head.atom_type is SymbolType and tree.head.value in self.unknowns
            ) or any(self.depends(child) for child in tree.args)
        return self._dependencies[tree]

    def known(self, tree: Tree, *, expand: bool = False) -> ast.expr:
        parameters = self.parameters

        class Names(ast.NodeTransformer):
            def visit_Name(self, node: ast.Name) -> ast.Name:
                return ast.Name(id=mangle_name(node.id), ctx=ast.Load()) if node.id in parameters else node

        value = typing.cast("ast.expr", Names().visit(expression_ast(tree)))
        return _expand(value) if expand else value

    def split(self, tree: Tree) -> OperatorParts:
        if not self.depends(tree):
            value = self.known(tree)
            return OperatorParts(_expand(_zero(value)), value)
        if tree.is_leaf:
            index = self.unknowns.index(typing.cast("str", tree.head.value))
            return OperatorParts(_item(self.basis_name, index), _item(self.zero_name, index))
        if tree.head is MUL:
            return self.product(tree)
        if tree.head in (ADD, NEG):
            children = [self.split(child) for child in tree.args]
        elif tree.head.atom_type in (ReductionOpType, IndexOpType):
            children = [self.split(tree.args[0])]
        else:
            message = f"Unsupported unknown-dependent linear operator during lowering: {tree.head}."
            raise BiltLinearBlockError(message)
        return OperatorParts(
            _apply(tree, [child.linear for child in children]),
            _apply(tree, [child.constant for child in children]),
        )

    def product(self, tree: Tree) -> OperatorParts:
        dependent = [index for index, child in enumerate(tree.args) if self.depends(child)]
        if len(dependent) != 1:
            message = "A linear product must have exactly one unknown-dependent factor."
            raise BiltLinearBlockError(message)
        index = dependent[0]
        value = self.split(tree.args[index])
        linear = [value.linear if i == index else self.known(child, expand=True) for i, child in enumerate(tree.args)]
        constant = [value.constant if i == index else self.known(child) for i, child in enumerate(tree.args)]
        return OperatorParts(_apply(tree, linear), _apply(tree, constant))


def _expand(value: ast.expr) -> ast.Call:
    return ast.Call(func=ast.Name(id=LINEAR_EXPAND_INTRINSIC_NAME, ctx=ast.Load()), args=[value], keywords=[])


def _item(name: str, index: int) -> ast.Subscript:
    return ast.Subscript(value=ast.Name(id=name, ctx=ast.Load()), slice=ast.Constant(index), ctx=ast.Load())


def _apply(tree: Tree, operands: list[ast.expr]) -> ast.expr:
    """Use existing operator code generation for the supplied value operands."""
    replacements = {f"__bilt_operand_{i}": value for i, value in enumerate(operands)}
    template = Tree(tree.head, tuple(symbol_tree(name) for name in replacements) + tree.args[len(operands) :])

    class Operands(ast.NodeTransformer):
        def visit_Name(self, node: ast.Name) -> ast.expr:
            return replacements.get(node.id, node)

    return typing.cast("ast.expr", Operands().visit(expression_ast(template)))


def _zero(value: ast.expr, *, basis: bool = False) -> ast.Call:
    return ast.Call(
        func=ast.Name(id=LINEAR_ZERO_INTRINSIC_NAME, ctx=ast.Load()),
        args=[value],
        keywords=[ast.keyword(arg="basis", value=ast.Constant(value=basis))],
    )

"""Small backend-neutral, SSA-style IR for realised equation expressions.

The IR stores operations, operands, attributes and nested control-flow regions,
not Python AST nodes. It deliberately does not replace symbolic Tree: Tree
represents an undirected relationship; FunctionIR represents a selected
calculation after isolation and domain-aware expression lowering.
"""

import ast
import dataclasses
import itertools
import typing

from bilt.core._to_ast import (
    MASK_AND_INTRINSIC_NAME,
    MASK_NOT_INTRINSIC_NAME,
    MASK_OR_INTRINSIC_NAME,
    MAX_INTRINSIC_NAME,
    MEAN_INTRINSIC_NAME,
    MIN_INTRINSIC_NAME,
    PROD_INTRINSIC_NAME,
    SUM_INTRINSIC_NAME,
    WHERE_INTRINSIC_NAME,
    isel_intrinsic_ast,
    reduction_intrinsic_ast,
)
from bilt.symbolic import _atoms as ops
from bilt.symbolic._atoms import AxisType, FloatType, IntType, QuantityType, SymbolType
from bilt.symbolic._isolation._errors import BiltUnsupportedIsolationError

if typing.TYPE_CHECKING:
    from bilt.symbolic._domains import Domain
    from bilt.symbolic._tree import Tree

__all__ = ["Block", "FunctionIR", "Operation", "Value", "ValueType", "ir_to_functiondef", "tree_to_ir"]


@dataclasses.dataclass(frozen=True, slots=True)
class Value:
    """An SSA result (integer) or a function argument (string)."""

    key: int | str


@dataclasses.dataclass(frozen=True, slots=True)
class ValueType:
    """Best-known numerical kind and named tensor domain of an SSA value."""

    kind: typing.Literal["number", "boolean", "unknown"] = "unknown"
    domain: Domain | None = None


@dataclasses.dataclass(frozen=True, slots=True)
class Operation:
    """One pure expression operation, optionally owning control-flow regions."""

    result: Value
    opcode: str
    operands: tuple[Value, ...] = ()
    attributes: tuple[tuple[str, int | float | bool], ...] = ()
    regions: tuple[Block, ...] = ()
    result_type: ValueType | None = None


@dataclasses.dataclass(frozen=True, slots=True)
class Block:
    """Ordered operations yielding one value; nested blocks are lazy regions."""

    operations: tuple[Operation, ...]
    yield_value: Value


@dataclasses.dataclass(frozen=True, slots=True)
class FunctionIR:
    """A realised function with its named tensor output domain."""

    name: str
    parameters: tuple[str, ...]
    body: Block
    output_domain: Domain | None = None
    parameter_types: tuple[tuple[str, ValueType], ...] = ()


class _Builder:
    def __init__(self, counter: itertools.count) -> None:
        self.counter = counter
        self.operations: list[Operation] = []

    def emit(
        self,
        opcode: str,
        operands: tuple[Value, ...] = (),
        *,
        attributes: tuple[tuple[str, int | float | bool], ...] = (),
        regions: tuple[Block, ...] = (),
    ) -> Value:
        result = Value(next(self.counter))
        kind: typing.Literal["number", "boolean", "unknown"] = "unknown"
        if opcode.startswith(("arith.", "bilt.sum", "bilt.mean", "bilt.min", "bilt.max", "bilt.prod", "bilt.isel")):
            kind = "number"
        elif opcode.startswith("cmp.") or opcode.startswith("bilt.mask_"):
            kind = "boolean"
        self.operations.append(Operation(result, opcode, operands, attributes, regions, ValueType(kind)))
        return result

    def finish(self, value: Value) -> Block:
        return Block(tuple(self.operations), value)


_BINARY_ARITY = 2

_BINARY_HEADS = {
    ops.POW: "arith.pow",
    ops.EQUALS: "cmp.eq",
    ops.NOT_EQUALS: "cmp.ne",
    ops.LESS_THAN: "cmp.lt",
    ops.LESS_THAN_OR_EQUALS: "cmp.le",
    ops.GREATER_THAN: "cmp.gt",
    ops.GREATER_THAN_OR_EQUALS: "cmp.ge",
}

_INTRINSIC_HEADS = {
    ops.WHERE: "bilt.where",
    ops.MASK_AND: "bilt.mask_and",
    ops.MASK_OR: "bilt.mask_or",
    ops.MASK_NOT: "bilt.mask_not",
}

_REDUCTION_HEADS = {
    ops.MEAN: "bilt.mean",
    ops.SUM: "bilt.sum",
    ops.MIN: "bilt.min",
    ops.MAX: "bilt.max",
    ops.PROD: "bilt.prod",
}


def tree_to_ir(
    tree: Tree,
    *,
    name: str,
    parameter_order: tuple[str, ...],
    output_domain: Domain | None = None,
    input_domains: typing.Mapping[str, Domain] | None = None,
) -> FunctionIR:
    """Convert an already domain-lowered expression directly into neutral IR."""
    builder = _Builder(itertools.count())
    value = _emit_tree(tree, builder)
    return FunctionIR(
        name=name,
        parameters=tuple(parameter for parameter in parameter_order if parameter != name),
        body=builder.finish(value),
        output_domain=output_domain,
        parameter_types=tuple(
            (parameter, ValueType(domain=input_domains[parameter]))
            for parameter in parameter_order
            if parameter != name and input_domains is not None and parameter in input_domains
        ),
    )


def _emit_tree(tree: Tree, builder: _Builder) -> Value:  # noqa: C901, PLR0911, PLR0912
    if tree.is_leaf:
        if tree.head.atom_type is SymbolType:
            return Value(typing.cast("str", tree.head.value))
        if tree.head.atom_type in (IntType, FloatType):
            return builder.emit("arith.constant", attributes=(("value", typing.cast("int | float", tree.head.value)),))
        if tree.head.atom_type is QuantityType:
            value = typing.cast("tuple[float, object, bool]", tree.head.value)[0]
            return builder.emit("arith.constant", attributes=(("value", value),))
        msg = f"Unsupported symbolic leaf {tree.head!r} in IR conversion."
        raise BiltUnsupportedIsolationError(msg)

    if tree.head is ops.CONDITIONAL:
        predicate, when_true, when_false = tree.args
        condition = _emit_tree(predicate, builder)
        true_builder = _Builder(builder.counter)
        false_builder = _Builder(builder.counter)
        true_result = _emit_tree(when_true, true_builder)
        false_result = _emit_tree(when_false, false_builder)
        return builder.emit(
            "scf.if",
            (condition,),
            regions=(true_builder.finish(true_result), false_builder.finish(false_result)),
        )

    if tree.head in (ops.ADD, ops.MUL):
        if not tree.args:
            msg = "Cannot lower an empty n-ary symbolic operator."
            raise BiltUnsupportedIsolationError(msg)
        children = iter(tree.args)
        value = _emit_tree(next(children), builder)
        for child in children:
            value = builder.emit(
                "arith.add" if tree.head is ops.ADD else "arith.mul",
                (value, _emit_tree(child, builder)),
            )
        return value

    if tree.head is ops.NEG:
        return builder.emit("arith.neg", (_emit_tree(tree.args[0], builder),))

    if tree.head is ops.INV:
        return builder.emit("arith.inv", (_emit_tree(tree.args[0], builder),))

    if tree.head in _BINARY_HEADS:
        if len(tree.args) != _BINARY_ARITY:
            msg = f"Unsupported symbolic predicate {tree!s}."
            raise BiltUnsupportedIsolationError(msg)
        return builder.emit(_BINARY_HEADS[tree.head], tuple(_emit_tree(arg, builder) for arg in tree.args))

    if tree.head in _INTRINSIC_HEADS:
        return builder.emit(_INTRINSIC_HEADS[tree.head], tuple(_emit_tree(arg, builder) for arg in tree.args))

    if tree.head in _REDUCTION_HEADS:
        value, axis_tree = tree.args
        axis = _positional_axis(axis_tree, "reduction")
        return builder.emit(
            _REDUCTION_HEADS[tree.head],
            (_emit_tree(value, builder),),
            attributes=(("axis", axis), ("keepdims", True)),
        )

    if tree.head is ops.ISEL:
        value, axis_tree, selector = tree.args
        axis = _positional_axis(axis_tree, "isel")
        return builder.emit(
            "bilt.isel",
            (_emit_tree(value, builder), _emit_tree(selector, builder)),
            attributes=(("axis", axis),),
        )

    msg = f"Unsupported symbolic head {tree.head!r} in IR conversion."
    raise BiltUnsupportedIsolationError(msg)


def _positional_axis(tree: Tree, operation: str) -> int:
    if not tree.is_leaf or tree.head.atom_type is not AxisType:
        msg = f"Lowered {operation} requires a positional axis."
        raise BiltUnsupportedIsolationError(msg)
    return typing.cast("int", tree.head.value)


_UNARY_AST = {"arith.neg": ast.USub, "arith.pos": ast.UAdd}
_BINARY_AST = {"arith.add": ast.Add, "arith.sub": ast.Sub, "arith.mul": ast.Mult, "arith.div": ast.Div, "arith.pow": ast.Pow}
_COMPARISON_AST = {
    "cmp.eq": ast.Eq,
    "cmp.ne": ast.NotEq,
    "cmp.lt": ast.Lt,
    "cmp.le": ast.LtE,
    "cmp.gt": ast.Gt,
    "cmp.ge": ast.GtE,
}
_INTRINSIC_NAMES = {
    "bilt.where": WHERE_INTRINSIC_NAME,
    "bilt.mask_and": MASK_AND_INTRINSIC_NAME,
    "bilt.mask_or": MASK_OR_INTRINSIC_NAME,
    "bilt.mask_not": MASK_NOT_INTRINSIC_NAME,
}
_REDUCTION_NAMES = {
    "bilt.mean": MEAN_INTRINSIC_NAME,
    "bilt.sum": SUM_INTRINSIC_NAME,
    "bilt.min": MIN_INTRINSIC_NAME,
    "bilt.max": MAX_INTRINSIC_NAME,
    "bilt.prod": PROD_INTRINSIC_NAME,
}


def ir_to_functiondef(function: FunctionIR) -> ast.FunctionDef:
    """Lower neutral IR to the existing Python AST backend interface.

    JAX uses its existing AST validation, conditional rewriting and JIT path
    after this pass; neither backend has to understand the IR in this prototype.
    """
    from bilt.core._ir_validation import validate_function_ir  # noqa: PLC0415

    validate_function_ir(function)
    body = _block_statements(function.body, function.parameters)
    function_def = ast.FunctionDef(
        name=function.name,
        args=ast.arguments(
            posonlyargs=[],
            args=[ast.arg(arg=parameter, annotation=None) for parameter in function.parameters],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[],
            vararg=None,
            kwarg=None,
        ),
        body=body,
        decorator_list=[],
        returns=None,
        type_comment=None,
    )
    ast.fix_missing_locations(function_def)
    return function_def


def _block_statements(block: Block, parameters: tuple[str, ...]) -> list[ast.stmt]:
    """Emit SSA bindings explicitly so order and shared subexpressions survive."""
    expressions: dict[Value, ast.expr] = {
        Value(parameter): ast.Name(id=parameter, ctx=ast.Load()) for parameter in parameters
    }
    reserved = set(parameters)
    statements: list[ast.stmt] = []
    for operation in block.operations:
        expression = _block_expression(Block((operation,), operation.result), expressions)
        temporary = f"__bilt_ir_{operation.result.key}"
        while temporary in reserved:
            temporary += "_"
        reserved.add(temporary)
        statements.append(ast.Assign(targets=[ast.Name(id=temporary, ctx=ast.Store())], value=expression))
        expressions[operation.result] = ast.Name(id=temporary, ctx=ast.Load())
    statements.append(ast.Return(value=expressions[block.yield_value]))
    return statements


def _block_expression(block: Block, visible: dict[Value, ast.expr]) -> ast.expr:  # noqa: C901
    expressions = visible.copy()
    for operation in block.operations:
        args = tuple(expressions[operand] for operand in operation.operands)
        attrs = dict(operation.attributes)
        name = operation.opcode
        if name == "arith.constant":
            value: ast.expr = ast.Constant(attrs["value"])
        elif name in _UNARY_AST:
            value = ast.UnaryOp(op=_UNARY_AST[name](), operand=args[0])
        elif name in _BINARY_AST:
            value = ast.BinOp(left=args[0], op=_BINARY_AST[name](), right=args[1])
        elif name == "arith.inv":
            value = ast.BinOp(left=ast.Constant(1), op=ast.Div(), right=args[0])
        elif name in _COMPARISON_AST:
            value = ast.Compare(left=args[0], ops=[_COMPARISON_AST[name]()], comparators=[args[1]])
        elif name in _INTRINSIC_NAMES:
            value = ast.Call(func=ast.Name(id=_INTRINSIC_NAMES[name], ctx=ast.Load()), args=list(args), keywords=[])
        elif name in _REDUCTION_NAMES:
            value = reduction_intrinsic_ast(_REDUCTION_NAMES[name], args[0], typing.cast("int", attrs["axis"]))
        elif name == "bilt.isel":
            value = isel_intrinsic_ast(args[0], args[1], typing.cast("int", attrs["axis"]))
        elif name == "scf.if":
            true_region, false_region = operation.regions
            value = ast.IfExp(
                test=args[0],
                body=_block_expression(true_region, expressions),
                orelse=_block_expression(false_region, expressions),
            )
        else:
            msg = f"Unsupported IR operation {name!r}."
            raise ValueError(msg)
        expressions[operation.result] = value
    return expressions[block.yield_value]

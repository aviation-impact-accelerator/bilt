"""Conservative Python AST to neutral IR lowering for pure transforms.

Unsupported source is left on the existing AST compilation path. This adapter
does not turn arbitrary Python or captured closures into backend-neutral code.
"""

import ast
import itertools
import typing

from bilt.core._ir import Block, FunctionIR, Operation, Value, ValueType

if typing.TYPE_CHECKING:
    from bilt.symbolic._domains import Domain


_BINARY = {
    ast.Add: "arith.add",
    ast.Sub: "arith.sub",
    ast.Mult: "arith.mul",
    ast.Div: "arith.div",
    ast.Pow: "arith.pow",
}
_UNARY = {ast.USub: "arith.neg", ast.UAdd: "arith.pos"}
_COMPARE = {
    ast.Eq: "cmp.eq",
    ast.NotEq: "cmp.ne",
    ast.Lt: "cmp.lt",
    ast.LtE: "cmp.le",
    ast.Gt: "cmp.gt",
    ast.GtE: "cmp.ge",
}
_INTRINSICS = {
    "__bilt_where__": ("bilt.where", 3),
    "__bilt_mask_and__": ("bilt.mask_and", 2),
    "__bilt_mask_or__": ("bilt.mask_or", 2),
    "__bilt_mask_not__": ("bilt.mask_not", 1),
    "__bilt_mean__": ("bilt.mean", 1),
    "__bilt_sum__": ("bilt.sum", 1),
    "__bilt_min__": ("bilt.min", 1),
    "__bilt_max__": ("bilt.max", 1),
    "__bilt_prod__": ("bilt.prod", 1),
    "__bilt_isel__": ("bilt.isel", 2),
}


class _UnsupportedIR(ValueError):
    """AST construct is deliberately outside the neutral subset."""


class _FunctionBuilder:
    def __init__(self) -> None:
        self.ids = itertools.count()
        self.operations: list[Operation] = []

    def emit(
        self,
        opcode: str,
        operands: tuple[Value, ...] = (),
        attributes: tuple[tuple[str, int | float | bool], ...] = (),
        regions: tuple[Block, ...] = (),
    ) -> Value:
        value = Value(next(self.ids))
        self.operations.append(Operation(value, opcode, operands, attributes, regions))
        return value

    def block(self, result: Value) -> Block:
        return Block(tuple(self.operations), result)


def ast_to_ir(function: ast.FunctionDef, *, output_domain: Domain | None = None) -> FunctionIR | None:
    """Lower supported side-effect-free transforms, otherwise return None."""
    if (
        getattr(function, "_bilt_runtime_bindings", None)
        or function.args.posonlyargs
        or function.args.kwonlyargs
        or function.args.vararg
        or function.args.kwarg
        or function.args.defaults
        or function.args.kw_defaults
    ):
        return None
    parameters = tuple(arg.arg for arg in function.args.args)
    if len(parameters) != len(set(parameters)):
        return None
    builder = _FunctionBuilder()
    names = {name: Value(name) for name in parameters}
    try:
        for statement in function.body[:-1]:
            if not (
                isinstance(statement, ast.Assign)
                and len(statement.targets) == 1
                and isinstance(statement.targets[0], ast.Name)
            ):
                return None
            names[statement.targets[0].id] = _expr(statement.value, builder, names)
        if not function.body or not isinstance(function.body[-1], ast.Return):
            return None
        returned = function.body[-1].value
        if returned is None:
            return None
        value = _expr(returned, builder, names)
    except _UnsupportedIR:
        return None
    return FunctionIR(
        name=function.name,
        parameters=parameters,
        body=builder.block(value),
        output_domain=output_domain,
    )


def _expr(  # noqa: C901, PLR0911
    node: ast.expr, builder: _FunctionBuilder, names: dict[str, Value]
) -> Value:
    if isinstance(node, ast.Name):
        try:
            return names[node.id]
        except KeyError:
            raise _UnsupportedIR(f"Unbound name {node.id!r}") from None
    if isinstance(node, ast.Constant) and type(node.value) in (int, float, bool):
        return builder.emit("arith.constant", attributes=(("value", node.value),))
    if isinstance(node, ast.BinOp):
        op = _BINARY.get(type(node.op))
        if op is None:
            raise _UnsupportedIR("Unsupported binary operation")
        return builder.emit(op, (_expr(node.left, builder, names), _expr(node.right, builder, names)))
    if isinstance(node, ast.UnaryOp):
        op = _UNARY.get(type(node.op))
        if op is None:
            raise _UnsupportedIR("Unsupported unary operation")
        return builder.emit(op, (_expr(node.operand, builder, names),))
    if isinstance(node, ast.Compare) and len(node.ops) == 1 and len(node.comparators) == 1:
        op = _COMPARE.get(type(node.ops[0]))
        if op is None:
            raise _UnsupportedIR("Unsupported comparison")
        return builder.emit(op, (_expr(node.left, builder, names), _expr(node.comparators[0], builder, names)))
    if isinstance(node, ast.IfExp):
        predicate = _expr(node.test, builder, names)
        when_true = _FunctionBuilder()
        when_true.ids = builder.ids
        when_false = _FunctionBuilder()
        when_false.ids = builder.ids
        yes = _expr(node.body, when_true, names)
        no = _expr(node.orelse, when_false, names)
        return builder.emit("scf.if", (predicate,), regions=(when_true.block(yes), when_false.block(no)))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        try:
            opcode, arity = _INTRINSICS[node.func.id]
        except KeyError:
            raise _UnsupportedIR("Unsupported call") from None
        if len(node.args) != arity:
            raise _UnsupportedIR("Invalid intrinsic argument count")
        attrs: list[tuple[str, int | float | bool]] = []
        for kw in node.keywords:
            if kw.arg not in ("axis", "keepdims") or not isinstance(kw.value, ast.Constant):
                raise _UnsupportedIR("Unsupported intrinsic keyword")
            value = kw.value.value
            if type(value) not in (int, float, bool):
                raise _UnsupportedIR("Unsupported intrinsic literal")
            attrs.append((kw.arg, value))
        if opcode.startswith("bilt.") and opcode not in (
            "bilt.where", "bilt.mask_and", "bilt.mask_or", "bilt.mask_not"
        ) and "axis" not in dict(attrs):
            raise _UnsupportedIR("Positional tensor axis missing")
        return builder.emit(opcode, tuple(_expr(arg, builder, names) for arg in node.args), tuple(attrs))
    raise _UnsupportedIR(f"Unsupported expression {type(node).__name__}")

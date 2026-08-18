"""Checks an AST for unsupported nodes. Raises an exception if any unsupported nodes are found."""

import ast

from bilt._backend._errors import UnsupportedNodeError

__all__ = ()


_SUPPORTED_NODES = frozenset(
    {
        ast.FunctionDef,
        ast.arguments,
        ast.arg,
        ast.Return,
        ast.Expr,
        ast.BinOp,
        ast.UnaryOp,
        ast.BoolOp,
        ast.Compare,
        ast.Call,
        ast.Constant,
        ast.Name,
        ast.Attribute,
        ast.Tuple,
        ast.List,
        ast.keyword,
        ast.Subscript,
        ast.Slice,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.FloorDiv,
        ast.Mod,
        ast.Pow,
        ast.MatMult,
        ast.UAdd,
        ast.USub,
        ast.Not,
        ast.Invert,
        ast.BitAnd,
        ast.BitOr,
        ast.BitXor,
        ast.LShift,
        ast.RShift,
        ast.And,
        ast.Or,
        ast.Eq,
        ast.NotEq,
        ast.Lt,
        ast.LtE,
        ast.Gt,
        ast.GtE,
        ast.Is,
        ast.IsNot,
        ast.In,
        ast.NotIn,
        ast.Load,
        ast.Store,
        ast.Del,
        ast.Assign,
    }
)


def _validate(fdef: ast.FunctionDef) -> None:
    for node in ast.walk(fdef):
        if type(node) not in _SUPPORTED_NODES:
            lineno = getattr(node, "lineno", "?")
            msg = f"AST node '{type(node).__name__}' at line {lineno} is unsupported by the Bilt Jax Backend."
            raise UnsupportedNodeError(msg)

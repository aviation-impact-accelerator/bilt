from __future__ import annotations
from .expressions import Binary, Constant, Expr, Symbol
class CannotIsolate(ValueError): pass

def isolate(expr: Expr, other: Expr, unknown: str) -> Expr:
    if isinstance(expr, Symbol) and expr.name == unknown: return other
    if not isinstance(expr, Binary): raise CannotIsolate(f"Cannot isolate {unknown!r} through {expr!s}")
    lh, rh = expr.left.contains(unknown), expr.right.contains(unknown)
    if lh and rh: raise CannotIsolate(f"Cannot isolate {unknown!r}: it appears on both sides of {expr!s}")
    if not lh and not rh: raise CannotIsolate(f"{unknown!r} not found in {expr!s}")
    l, r = expr.left, expr.right
    if expr.op == "+": return isolate(l, other - r, unknown) if lh else isolate(r, other - l, unknown)
    if expr.op == "-": return isolate(l, other + r, unknown) if lh else isolate(r, l - other, unknown)
    if expr.op == "*": return isolate(l, other / r, unknown) if lh else isolate(r, other / l, unknown)
    if expr.op == "/": return isolate(l, other * r, unknown) if lh else isolate(r, l / other, unknown)
    if expr.op == "**":
        if lh:
            if not isinstance(r, Constant): raise CannotIsolate("Cannot isolate base with non-constant exponent.")
            return isolate(l, other ** (Constant(1.0) / r), unknown)
        raise CannotIsolate("Cannot isolate an unknown exponent.")
    raise CannotIsolate(f"Unknown operator {expr.op!r}")

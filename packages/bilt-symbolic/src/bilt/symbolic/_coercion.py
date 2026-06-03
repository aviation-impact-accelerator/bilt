from bilt.symbolic._expressions import Expr


def as_expr(value: object) -> Expr:
    if isinstance(value, Expr):
        return value

    if isinstance(value, bool):
        msg = "Cannot convert bool to Expr."
        raise TypeError(msg)

    if isinstance(value, int):
        return Expr.integer(value)

    if isinstance(value, float):
        return Expr.real(value)

    msg = f"Cannot convert {type(value).__name__} to Expr."
    raise TypeError(msg)

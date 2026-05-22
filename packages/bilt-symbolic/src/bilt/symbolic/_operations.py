from bilt.symbolic._expressions import (
    Expr,
    contains_expression,
    evaluate_expression,
    expression_symbols,
    parse_expression,
)

type ValueEnv = dict[str, float]

type Evaluable = Expr


def contains(obj: Evaluable, symbol_name: str) -> bool:
    return contains_expression(obj, symbol_name)


def contains_symbols(obj: Evaluable) -> set[str]:
    return expression_symbols(obj)


def evaluate(obj: Evaluable, values: ValueEnv) -> float:
    return evaluate_expression(obj, values)


def parse(source: str) -> Evaluable:
    return parse_expression(source)

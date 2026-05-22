import ast
from dataclasses import dataclass
from typing import Literal

type Expr = Constant | Symbol | Binary
type ExprLike = Expr | float | int
type ValueEnv = dict[str, float]
type BinaryOperator = Literal["+", "-", "*", "/", "**"]


@dataclass(frozen=True, slots=True)
class Constant:
    value: float

    def __str__(self) -> str:
        return str(self.value)


@dataclass(frozen=True, slots=True)
class Symbol:
    name: str
    display_name: str | None = None

    def __str__(self) -> str:
        return self.display_name or self.name


@dataclass(frozen=True, slots=True)
class Binary:
    op: BinaryOperator
    left: Expr
    right: Expr

    def __str__(self) -> str:
        return f"({self.left} {self.op} {self.right})"


def as_expr(value: ExprLike) -> Expr:
    match value:
        case Constant() | Symbol() | Binary():
            return value

        case bool():
            message = "Boolean values are not supported as expressions"
            raise TypeError(message)

        case int() | float():
            return Constant(float(value))

        case _:
            message = f"Unsupported expression value: {type(value).__name__}"
            raise TypeError(message)


def binary(op: BinaryOperator, left: ExprLike, right: ExprLike) -> Binary:
    return Binary(op, as_expr(left), as_expr(right))


def add(left: ExprLike, right: ExprLike) -> Binary:
    return binary("+", left, right)


def sub(left: ExprLike, right: ExprLike) -> Binary:
    return binary("-", left, right)


def mul(left: ExprLike, right: ExprLike) -> Binary:
    return binary("*", left, right)


def div(left: ExprLike, right: ExprLike) -> Binary:
    return binary("/", left, right)


def pow_(left: ExprLike, right: ExprLike) -> Binary:
    return binary("**", left, right)


def evaluate_expression(expr: Expr, values: ValueEnv) -> float:
    match expr:
        case Constant(value):
            return value

        case Symbol(name):
            try:
                return values[name]
            except KeyError:
                message = f"Missing value for symbol {name!r}"
                raise KeyError(message) from None

        case Binary(op, left, right):
            left_value = evaluate_expression(left, values)
            right_value = evaluate_expression(right, values)
            return evaluate_binary(op, left_value, right_value)


def evaluate_binary(op: str, left: float, right: float) -> float:
    match op:
        case "+":
            return left + right
        case "-":
            return left - right
        case "*":
            return left * right
        case "/":
            return left / right
        case "**":
            return left**right
        case _:
            msg = "Unsuppported arithmetic operator"
            raise SyntaxError(msg)

    message = f"Unknown operator {op!r}"
    raise ValueError(message)


def expression_symbols(expr: Expr) -> set[str]:
    match expr:
        case Constant():
            return set()

        case Symbol(name):
            return {name}

        case Binary(_, left, right):
            return expression_symbols(left) | expression_symbols(right)


def contains_expression(expr: Expr, symbol_name: str) -> bool:
    match expr:
        case Constant():
            return False

        case Symbol(name):
            return name == symbol_name

        case Binary(_, left, right):
            return contains_expression(left, symbol_name) or contains_expression(right, symbol_name)


def parse_expression(source: str) -> Expr:
    """Parse a small plain-Python expression into the BILT expression IR.

    Supported syntax:
    - numeric constants
    - symbol names
    - binary operations: +, -, *, /, **

    Unsupported for now:
    - unary operations, e.g. -x
    - function calls, e.g. sin(x)
    - attribute access, e.g. state.temperature
    - indexing, e.g. x[0]
    """
    tree = ast.parse(source, mode="eval")
    return lower_ast_expr(tree.body)


def lower_ast_expr(node: ast.expr) -> Expr:
    if isinstance(node, ast.Constant):
        return lower_ast_constant(node)

    if isinstance(node, ast.Name):
        return Symbol(node.id)

    if isinstance(node, ast.BinOp):
        return Binary(
            lower_ast_binary_operator(node.op),
            lower_ast_expr(node.left),
            lower_ast_expr(node.right),
        )

    message = f"Unsupported Python expression: {ast.dump(node)}"
    raise SyntaxError(message)


def lower_ast_constant(node: ast.Constant) -> Constant:
    value: object = node.value

    if isinstance(value, bool):
        message = "Boolean constants are not supported"
        raise TypeError(message)

    if isinstance(value, int | float):
        return Constant(float(value))

    message = f"Unsupported Python constant: {value!r}"
    raise SyntaxError(message)


def lower_ast_binary_operator(op: ast.operator) -> BinaryOperator:
    match op:
        case ast.Add():
            return "+"
        case ast.Sub():
            return "-"
        case ast.Mult():
            return "*"
        case ast.Div():
            return "/"
        case ast.Pow():
            return "**"
        case _:
            msg = "Unsupported Python binary operator"
            raise SyntaxError(msg)

    message = f"Unsupported Python binary operator: {type(op).__name__}"
    raise SyntaxError(message)

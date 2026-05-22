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


@dataclass(frozen=True, slots=True)
class Equality:
    lhs: Expr
    rhs: Expr

    def __str__(self) -> str:
        return f"{self.lhs} == {self.rhs}"


def as_expr(value: ExprLike) -> Expr:
    match value:
        case Constant() | Symbol() | Binary():
            return value

        case int() | float():
            return Constant(float(value))


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


def neg(expr: ExprLike) -> Binary:
    return mul(-1.0, expr)


def evaluate(expr: Expr, values: ValueEnv) -> float:
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
            left_value = evaluate(left, values)
            right_value = evaluate(right, values)
            return evaluate_binary(op, left_value, right_value)


def evaluate_binary(op: BinaryOperator, left: float, right: float) -> float:
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

    message = f"Unknown operator {op!r}"
    raise ValueError(message)


def symbols(expr: Expr) -> set[str]:
    match expr:
        case Constant():
            return set()

        case Symbol(name):
            return {name}

        case Binary(_, left, right):
            return symbols(left) | symbols(right)


def contains(expr: Expr, symbol_name: str) -> bool:
    match expr:
        case Constant():
            return False

        case Symbol(name):
            return name == symbol_name

        case Binary(_, left, right):
            return contains(left, symbol_name) or contains(right, symbol_name)


def residual(equality: Equality) -> Expr:
    return sub(equality.lhs, equality.rhs)


def parse_expr(source: str) -> Expr:
    """Parse a small plain-Python expression into the BILT expression IR."""
    tree = ast.parse(source, mode="eval")
    return lower_ast_expr(tree.body)


def parse_equality(source: str) -> Equality:
    """Parse a plain Python assignment into an Equality.

    Example:
        parse_equality("Re = rho * u * D / mu")
    """
    tree = ast.parse(source, mode="exec")

    if len(tree.body) != 1:
        message = "Expected exactly one statement"
        raise SyntaxError(message)

    statement = tree.body[0]

    match statement:
        case ast.Assign(targets=[lhs], value=rhs):
            return Equality(
                lhs=lower_ast_expr(lhs),
                rhs=lower_ast_expr(rhs),
            )

    message = "Expected an assignment such as 'Re = rho * u * D / mu'"
    raise SyntaxError(message)


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
    value = node.value

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

    message = f"Unsupported Python binary operator: {type(op).__name__}"
    raise SyntaxError(message)

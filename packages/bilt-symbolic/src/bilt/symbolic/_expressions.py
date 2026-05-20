from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

type ValueEnv = dict[str, float]
type ExprLike = Expr | float | int


def as_expr(value: ExprLike) -> Expr:
    if isinstance(value, Expr):
        return value

    return Constant(float(value))


class Expr(ABC):
    @abstractmethod
    def eval(self, values: ValueEnv) -> float:
        raise NotImplementedError

    @abstractmethod
    def contains(self, symbol_name: str) -> bool:
        raise NotImplementedError

    @abstractmethod
    def symbols(self) -> set[str]:
        raise NotImplementedError

    def __add__(self, other: ExprLike) -> Expr:
        return Binary("+", self, as_expr(other))

    def __radd__(self, other: ExprLike) -> Expr:
        return Binary("+", as_expr(other), self)

    def __sub__(self, other: ExprLike) -> Expr:
        return Binary("-", self, as_expr(other))

    def __rsub__(self, other: ExprLike) -> Expr:
        return Binary("-", as_expr(other), self)

    def __mul__(self, other: ExprLike) -> Expr:
        return Binary("*", self, as_expr(other))

    def __rmul__(self, other: ExprLike) -> Expr:
        return Binary("*", as_expr(other), self)

    def __truediv__(self, other: ExprLike) -> Expr:
        return Binary("/", self, as_expr(other))

    def __rtruediv__(self, other: ExprLike) -> Expr:
        return Binary("/", as_expr(other), self)

    def __pow__(self, other: ExprLike) -> Expr:
        return Binary("**", self, as_expr(other))

    def __rpow__(self, other: ExprLike) -> Expr:
        return Binary("**", as_expr(other), self)

    def __neg__(self) -> Expr:
        return Binary("*", Constant(-1.0), self)


@dataclass(frozen=True, slots=True)
class Constant(Expr):
    value: float

    def eval(self, values: ValueEnv) -> float:
        _ = values
        return self.value

    def contains(self, symbol_name: str) -> bool:
        _ = symbol_name
        return False

    def symbols(self) -> set[str]:
        return set()

    def __str__(self) -> str:
        return str(self.value)


@dataclass(frozen=True, slots=True)
class Symbol(Expr):
    name: str
    display_name: str | None = None

    def eval(self, values: ValueEnv) -> float:
        try:
            return values[self.name]
        except KeyError:
            message = f"Missing value for symbol {self.name!r}"
            raise KeyError(message) from None

    def contains(self, symbol_name: str) -> bool:
        return self.name == symbol_name

    def symbols(self) -> set[str]:
        return {self.name}

    def __str__(self) -> str:
        return self.display_name or self.name


@dataclass(frozen=True, slots=True)
class Binary(Expr):
    op: str
    left: Expr
    right: Expr

    def eval(self, values: ValueEnv) -> float:
        a = self.left.eval(values)
        b = self.right.eval(values)

        if self.op == "+":
            return a + b
        if self.op == "-":
            return a - b
        if self.op == "*":
            return a * b
        if self.op == "/":
            return a / b
        if self.op == "**":
            return a**b

        message = f"Unknown operator {self.op!r}"
        raise ValueError(message)

    def contains(self, symbol_name: str) -> bool:
        return self.left.contains(symbol_name) or self.right.contains(symbol_name)

    def symbols(self) -> set[str]:
        return self.left.symbols() | self.right.symbols()

    def __str__(self) -> str:
        return f"({self.left} {self.op} {self.right})"

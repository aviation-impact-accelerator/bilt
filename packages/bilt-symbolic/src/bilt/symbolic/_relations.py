from dataclasses import dataclass
from typing import Literal

from bilt.symbolic._expressions import Expr, ExprLike, ValueEnv, as_expr, evaluate, sub, symbols

type RelationOperator = Literal["==", "<=", ">="]
type RelationKind = Literal["equality", "less_equal", "greater_equal"]


@dataclass(frozen=True, slots=True)
class Relation:
    op: RelationOperator
    lhs: Expr
    rhs: Expr

    def __str__(self) -> str:
        return relation_to_string(self)


def relation(op: RelationOperator, lhs: ExprLike, rhs: ExprLike) -> Relation:
    return Relation(op, as_expr(lhs), as_expr(rhs))


def eq(lhs: ExprLike, rhs: ExprLike) -> Relation:
    return relation("==", lhs, rhs)


def le(lhs: ExprLike, rhs: ExprLike) -> Relation:
    return relation("<=", lhs, rhs)


def ge(lhs: ExprLike, rhs: ExprLike) -> Relation:
    return relation(">=", lhs, rhs)

Eq = eq
Le = le
Ge = ge


def relation_to_string(rel: Relation) -> str:
    return f"{rel.lhs} {rel.op} {rel.rhs}"


def relation_kind(rel: Relation) -> RelationKind:
    match rel.op:
        case "==":
            return "equality"
        case "<=":
            return "less_equal"
        case ">=":
            return "greater_equal"


def relation_residual(rel: Relation) -> Expr:
    return sub(rel.lhs, rel.rhs)


def relation_symbols(rel: Relation) -> set[str]:
    return symbols(rel.lhs) | symbols(rel.rhs)


def evaluate_relation(rel: Relation, values: ValueEnv) -> bool:
    lhs_value = evaluate(rel.lhs, values)
    rhs_value = evaluate(rel.rhs, values)

    match rel.op:
        case "==":
            return lhs_value == rhs_value
        case "<=":
            return lhs_value <= rhs_value
        case ">=":
            return lhs_value >= rhs_value
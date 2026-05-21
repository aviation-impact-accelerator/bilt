__all__ = [
    "Equation",
    "Relation",
    "RelationKind",
    "RelationOperator",
    "Symbol",
    "display_name_from_metadata",
    "eq",
    "equation",
    "equation_symbols",
    "ge",
    "le",
    "relation",
    "relation_to_string",
    "split_annotated",
    "symbol_from_parameter",
]
import typing
from dataclasses import dataclass
from inspect import Signature, signature
from typing import Annotated, Any, Literal, get_args, get_origin, get_type_hints

type RelationOperator = Literal["==", "<=", ">="]
type RelationKind = Literal["equality", "less_equal", "greater_equal"]

if typing.TYPE_CHECKING:
    from collections.abc import Callable


@dataclass(frozen=True, slots=True)
class Symbol:
    name: str
    display_name: str | None = None

    def __str__(self) -> str:
        return self.display_name or self.name


@dataclass(frozen=True, slots=True)
class Relation:
    op: RelationOperator
    lhs: object
    rhs: object

    def __str__(self) -> str:
        return relation_to_string(self)


def relation(
    op: RelationOperator,
    lhs: object,
    rhs: object,
) -> Relation:
    return Relation(op, lhs, rhs)


def relation_to_string(rel: Relation) -> str:
    return f"{rel.lhs} {rel.op} {rel.rhs}"


def eq(lhs: object, rhs: object) -> Relation:
    return relation("==", lhs, rhs)


def le(lhs: object, rhs: object) -> Relation:
    return relation("<=", lhs, rhs)


def ge(lhs: object, rhs: object) -> Relation:
    return relation(">=", lhs, rhs)


@dataclass(frozen=True, slots=True)
class Equation:
    name: str
    relation: Relation
    symbols: dict[str, Symbol]
    signature: Signature

    def __str__(self) -> str:
        return f"{self.name}: {self.relation}"


def equation(function: Callable[..., Relation]) -> Equation:
    function_signature = signature(function)
    function_name = getattr(function, "__name__", function.__class__.__name__)
    type_hints = get_type_hints(function, include_extras=True)

    symbols = equation_symbols(
        function_signature,
        type_hints,
    )

    relation_ = function(**symbols)

    if not isinstance(relation_, Relation):
        msg = f"Equation function {function_name!r} must return a Relation."
        raise TypeError(msg)

    return Equation(
        name=function_name,
        relation=relation_,
        symbols=symbols,
        signature=function_signature,
    )


def equation_symbols(
    function_signature: Signature,
    type_hints: dict[str, object],
) -> dict[str, Symbol]:
    return {name: symbol_from_parameter(name, type_hints.get(name, Any)) for name in function_signature.parameters}


def symbol_from_parameter(name: str, annotation: object) -> Symbol:
    _base_annotation, metadata = split_annotated(annotation)

    for item in metadata:
        if isinstance(item, Symbol):
            return Symbol(
                name=name,
                display_name=item.display_name or item.name,
            )

    return Symbol(
        name=name,
        display_name=display_name_from_metadata(metadata),
    )


def split_annotated(annotation: object) -> tuple[object, tuple[object, ...]]:
    if get_origin(annotation) is Annotated:
        args = get_args(annotation)
        return args[0], tuple(args[1:])

    return annotation, ()


def display_name_from_metadata(metadata: tuple[object, ...]) -> str | None:
    for item in metadata:
        value = getattr(item, "display_name", None)

        if isinstance(value, str):
            return value

        value = getattr(item, "name", None)

        if isinstance(value, str):
            return value

    return None

from __future__ import annotations
from inspect import signature
from typing import Any, Callable, Annotated, get_args, get_origin, get_type_hints
from examples.symbolic_api_and_basic_backend.annotations import Quantity, SymbolMetadata
from examples.symbolic_api_and_basic_backend.equations import Equation
from examples.symbolic_api_and_basic_backend.expressions import Symbol
from examples.symbolic_api_and_basic_backend.relations import Relation

def is_quantity_annotation_type(base_type: Any) -> bool:
    return base_type is Quantity or getattr(base_type, "__name__", None) == "Quantity"

def parse_quantity_annotation(annotation: Any) -> tuple[Any, SymbolMetadata | None]:
    if get_origin(annotation) is not Annotated: raise TypeError("Equation parameters must be annotated as Annotated[Quantity, unit].")
    args = get_args(annotation)
    if len(args) < 2: raise TypeError("Annotated[Quantity, unit] requires a unit metadata item.")
    if not is_quantity_annotation_type(args[0]): raise TypeError(f"Equation parameters must use Annotated[Quantity, unit], got base type {args[0]!r}.")
    metadata = next((item for item in args[2:] if isinstance(item, SymbolMetadata)), None)
    return args[1], metadata

def equation(fn: Callable[..., Relation]) -> Equation:
    sig = signature(fn); type_hints = get_type_hints(fn, include_extras=True)
    symbol_table: dict[str, Symbol] = {}; symbolic_args: dict[str, Symbol] = {}
    for parameter_name, parameter in sig.parameters.items():
        unit, metadata = parse_quantity_annotation(type_hints.get(parameter_name, parameter.annotation))
        symbol = Symbol(parameter_name, unit, metadata.display_name if metadata else None)
        symbol_table[parameter_name] = symbol; symbolic_args[parameter_name] = symbol
    relation = fn(**symbolic_args)
    if isinstance(relation, bool): raise TypeError(f"{fn.__name__} returned a bool. Use Eq(lhs, rhs) if needed.")
    if not isinstance(relation, Relation): raise TypeError(f"{fn.__name__} must return a Relation, for example `return Eq(lhs, rhs)`.")
    eq = Equation(fn.__name__, relation, symbol_table, sig)
    eq.check_units()
    return eq

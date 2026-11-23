__all__ = [
    "BasicEquation",
    "Equation",
    "check_homogeneity",
    "equation",
]

import inspect
import types
import typing
lazy import enum

from bilt.core._dimensional_analysis import absoluteness_of, dimensionality_of
from bilt.core._enum_matching import enum_port_types, enum_tensor_ports
from bilt.core._equation_ast import BiltCategoricalEquationError
from bilt.core._equation_tracing import validated_symbolic_function
from bilt.core._errors import (
    BiltEquationDimensionalityError,
    BiltEquationReturnError,
    BiltEquationSignatureError,
    BiltUnresolvedAnnotationError,
)
from bilt.core._interface import Interface
from bilt.core._port import QualifiedPort
from bilt.core._port_declaration import PortDeclaration, annotated_parts, port_declarations, quantity_magnitude_kind
from bilt.core._sub_system import qualified_enum_ports, validate_name
from bilt.core._tensors._tensors import QuantityTensor, Tensor
from bilt.core._validation import validate_independently
from bilt.symbolic import _atoms as ops
from bilt.symbolic._atoms import EnumType, QuantityType, SymbolType
from bilt.symbolic._expressions import Expr, SubSystemExpr, symbol_expr, tensor_symbol_expr
from bilt.symbolic._tree import Tree, symbol_tree
from bilt.symbolic._walk import subtrees
from bilt.units._errors import BiltDimensionalityError

if typing.TYPE_CHECKING:
    import collections.abc

    from bilt.units import Dimension


class Equation[R, **P](typing.Protocol):
    __bilt_name__: str
    __bilt_parameters__: tuple[str, ...]
    __bilt_port_declarations__: dict[str, PortDeclaration]
    __bilt_equation_symbols__: dict[str, Expr]
    __bilt_equation_tree__: Tree
    __bilt_enum_ports__: dict[str, type[enum.Enum]]
    __bilt_enum_tensor_ports__: frozenset[str]
    __bilt_sub_systems__: types.MappingProxyType[str, Interface]

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R: ...


class BasicEquation[R, **P]:
    __bilt_name__: str
    __bilt_parameters__: tuple[str, ...]
    __bilt_port_declarations__: dict[str, PortDeclaration]
    __bilt_equation_symbols__: dict[str, Expr]
    __bilt_equation_tree__: Tree
    __bilt_enum_ports__: dict[str, type[enum.Enum]]
    __bilt_enum_tensor_ports__: frozenset[str]
    __bilt_sub_systems__: types.MappingProxyType[str, Interface]
    _function: collections.abc.Callable[P, R]

    def __init__(self, function: collections.abc.Callable[P, R]) -> None:
        name = _function_name(function)
        try:
            signature = inspect.signature(function, eval_str=True)
        except NameError as error:
            raise BiltUnresolvedAnnotationError(name, error) from error

        _validate_equation_signature(name, signature)

        sub_systems = _sub_system_parameters(signature)
        annotations = {port: parameter.annotation for port, parameter in signature.parameters.items()}
        enum_types = enum_port_types(name, annotations)
        tensor_ports = enum_tensor_ports(annotations)
        sub_system_enum_types = {
            port: enum_type
            for child, interface in sub_systems.items()
            for port, enum_type in qualified_enum_ports(child, interface).items()
        }
        symbols = _build_symbols(signature, sub_systems)
        traced_function = validated_symbolic_function(function, enum_types=enum_types | sub_system_enum_types)
        tree = _build_equation_tree(name, traced_function, symbols)
        occurring = subtrees(tree)
        equation_symbols, referenced_sub_systems, sub_system_declarations = _expand_sub_system_parameters(
            symbols, sub_systems, occurring
        )

        referenced_enum_types = {
            port: enum_type for port, enum_type in sub_system_enum_types.items() if port in equation_symbols
        }
        self.__bilt_enum_ports__ = enum_types | referenced_enum_types
        self.__bilt_enum_tensor_ports__ = tensor_ports | frozenset(
            f"{child}.{port}"
            for child, interface in sub_systems.items()
            for port in interface.__bilt_enum_tensor_ports__
            if f"{child}.{port}" in equation_symbols
        )
        _validate_enum_uses(occurring, self.__bilt_enum_ports__)
        self.__bilt_name__ = name
        self.__bilt_parameters__ = tuple(equation_symbols)
        self.__bilt_port_declarations__ = (
            port_declarations(
                name,
                {
                    port: parameter.annotation
                    for port, parameter in signature.parameters.items()
                    if port not in sub_systems
                },
            )
            | sub_system_declarations
        )
        self.__bilt_equation_symbols__ = equation_symbols
        self.__bilt_equation_tree__ = tree
        self.__bilt_sub_systems__ = types.MappingProxyType(referenced_sub_systems)
        self._function = function
        dimensionalities = {
            port: declaration.dimensionality for port, declaration in self.__bilt_port_declarations__.items()
        }
        validate_independently(
            (
                lambda: check_homogeneity(self, dimensionalities),
                lambda: _carried_under(name, tree, dimensionalities),
                lambda: absoluteness_of(name, tree, self.__bilt_port_declarations__),
            )
        )

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        return self._function(*args, **kwargs)

    def __str__(self) -> str:
        return f"{self.__bilt_name__}: {self.__bilt_equation_tree__}"


def _function_name(function: collections.abc.Callable[..., object]) -> str:
    return getattr(
        function,
        "__name__",
        type(function).__name__,
    )


def _validate_equation_signature(
    name: str,
    signature: inspect.Signature,
) -> None:
    for param in signature.parameters.values():
        validate_name(param.name)

        if param.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            msg = (
                f"Equation function {name!r} must use only named parameters "
                "(no positional-only parameters, *args, or **kwargs)."
            )
            raise BiltEquationSignatureError(msg)


def _sub_system_parameters(signature: inspect.Signature) -> dict[str, Interface]:
    """Pick out the parameters annotated with a system interface, which stand for sub-systems."""
    return {
        name: parameter.annotation
        for name, parameter in signature.parameters.items()
        if isinstance(parameter.annotation, Interface)
    }


def _build_symbols(
    signature: inspect.Signature,
    sub_systems: collections.abc.Mapping[str, Interface],
) -> dict[str, Expr | SubSystemExpr]:
    """Bind each parameter to what the equation function sees when called symbolically.

    Tensor-specific annotations get tensor symbols for named reductions and
    selections. Pointwise scalar-or-tensor quantities use ordinary symbols,
    which Bilt can lift over compatible domains.
    """
    symbols: dict[str, Expr | SubSystemExpr] = {}

    for name, parameter in signature.parameters.items():
        base_type, _ = annotated_parts(parameter.annotation)
        if name in sub_systems:
            symbols[name] = SubSystemExpr(
                name,
                {port: QualifiedPort(name, port).dotted for port in sub_systems[name].__bilt_parameters__},
            )
        elif base_type in (Tensor, QuantityTensor) or quantity_magnitude_kind(base_type) == "tensor":
            symbols[name] = tensor_symbol_expr(name)
        else:
            symbols[name] = symbol_expr(name)

    return symbols


def _expand_sub_system_parameters(
    symbols: dict[str, Expr | SubSystemExpr],
    sub_systems: collections.abc.Mapping[str, Interface],
    occurring: collections.abc.Container[Tree],
) -> tuple[dict[str, Expr], dict[str, Interface], dict[str, PortDeclaration]]:
    """Replace each sub-system parameter with the dotted ports the equation actually references.

    Keys keep signature order, so the parameter tuple reads straight off the result. A port the
    equation never mentions is not a parameter of it, so a sub-system contributes only what it uses.
    Each dotted port carries the declaration its sub-system's interface gives it.
    """
    equation_symbols: dict[str, Expr] = {}
    referenced_sub_systems: dict[str, Interface] = {}
    sub_system_declarations: dict[str, PortDeclaration] = {}

    for name, symbol in symbols.items():
        if isinstance(symbol, Expr):
            equation_symbols[name] = symbol
            continue

        for port in sorted(sub_systems[name].__bilt_parameters__):
            dotted = QualifiedPort(name, port).dotted
            if symbol_tree(dotted) in occurring:
                equation_symbols[dotted] = symbol_expr(dotted)
                referenced_sub_systems[name] = sub_systems[name]
                if port in sub_systems[name].__bilt_port_declarations__:
                    sub_system_declarations[dotted] = sub_systems[name].__bilt_port_declarations__[port]

    return equation_symbols, referenced_sub_systems, sub_system_declarations


def _validate_enum_uses(occurring: set[Tree], enum_types: collections.abc.Mapping[str, type[enum.Enum]]) -> None:
    """Check categorical operands that were introduced by Python calls during tracing."""
    enum_symbols = {symbol_tree(port): enum_type for port, enum_type in enum_types.items()}
    categorical = {node for node in occurring if node.head.atom_type is EnumType or node in enum_symbols}
    if not categorical:
        return

    allowed_predicates = {
        node.args[0]
        for node in occurring
        if node.head is ops.CONDITIONAL and _is_enum_predicate(node.args[0], enum_symbols)
    }
    for node in occurring:
        for index, operand in enumerate(node.args):
            if operand in categorical and node not in allowed_predicates:
                if operand in enum_symbols:
                    msg = f"Enum port {operand.head.value!r}: enum parameters may only be used as match subjects."
                else:
                    enum_type, member = typing.cast("tuple[type[enum.Enum], str]", operand.head.value)
                    msg = f"Enum member {enum_type.__qualname__}.{member} may only be used in match case patterns."
                raise BiltCategoricalEquationError(msg)
            if operand in allowed_predicates and (node.head is not ops.CONDITIONAL or index != 0):
                msg = "Enum comparisons may only be used as match case predicates."
                raise BiltCategoricalEquationError(msg)


def _is_enum_predicate(predicate: Tree, enum_symbols: dict[Tree, type[enum.Enum]]) -> bool:
    if (
        not predicate.is_equality
        or predicate.left not in enum_symbols
        or predicate.right.head.atom_type is not EnumType
    ):
        return False
    enum_type, _ = typing.cast("tuple[type[enum.Enum], str]", predicate.right.head.value)
    return enum_type is enum_symbols[predicate.left]


def _build_equation_tree[R, **P](
    name: str,
    function: collections.abc.Callable[P, R],
    symbols: dict[str, Expr | SubSystemExpr],
) -> Tree:
    symbolic_function = typing.cast(
        "collections.abc.Callable[..., object]",
        function,
    )
    result = symbolic_function(**symbols)

    if not isinstance(result, Tree):
        raise BiltEquationReturnError(_equation_return_error_message(name, result))

    if not result.is_relation:
        raise BiltEquationReturnError(_equation_return_error_message(name, result))

    return result


def _equation_return_error_message(name: str, result: object) -> str:
    if isinstance(result, Expr):
        return (
            f"Equation {name!r} returned an expression, "
            "not an equation.\n\n"
            "Did you forget to compare it with another expression?\n\n"
            "Use:\n"
            "    return lhs == rhs"
        )

    if isinstance(result, Tree):
        return (
            f"Equation {name!r} returned a non-relation tree.\n\n"
            "Equation functions must return a relation tree.\n\n"
            "Use:\n"
            "    return lhs == rhs"
        )

    return (
        f"Equation {name!r} must return a relation tree, "
        f"but returned {type(result).__name__!r}.\n\n"
        "Use:\n"
        "    return lhs == rhs"
    )


def equation[R, **P](
    function: collections.abc.Callable[P, R],
) -> BasicEquation[R, P]:
    """Decorator to denote a function as an equation."""
    return BasicEquation(function)


def check_homogeneity(
    equation: BasicEquation[object, ...],
    dimensionalities: collections.abc.Mapping[str, Dimension],
) -> None:
    """Demand that an equation's relation can be dimensionally homogeneous, given its ports' `dimensionalities`."""
    tree = equation.__bilt_equation_tree__
    try:
        left = dimensionality_of(tree.left, dimensionalities)
        right = dimensionality_of(tree.right, dimensionalities)
    except BiltDimensionalityError as error:
        raise BiltEquationDimensionalityError(
            equation.__bilt_name__, f"is dimensionally inconsistent: {error}"
        ) from error

    if left is not None and right is not None and left is not right:
        msg = f"relates {left} to {right}"
        raise BiltEquationDimensionalityError(equation.__bilt_name__, msg)


def _carried_under(
    equation_name: str,
    tree: Tree,
    dimensionalities: collections.abc.Mapping[str, Dimension],
) -> tuple[frozenset[str], frozenset[Dimension]]:
    """Return the ports and the quantity constants' dimensionalities that `tree` carries.

    A power's exponent and an index operation's arguments do not carry what is under them up.
    """
    if tree.is_leaf:
        if tree.head.atom_type is SymbolType:
            return frozenset({typing.cast("str", tree.head.value)}), frozenset()
        if tree.head.atom_type is QuantityType:
            return frozenset(), frozenset({typing.cast("tuple[float, Dimension, bool]", tree.head.value)[1]})
        return frozenset(), frozenset()

    under = [_carried_under(equation_name, operand, dimensionalities) for operand in tree.args]
    if tree.head in (ops.MASK_AND, ops.MASK_OR, ops.MASK_NOT):
        # Comparisons have already been checked independently; mask composition
        # does not mix units or quantity kinds between predicates.
        return frozenset(), frozenset()
    if tree.head in (ops.CONDITIONAL, ops.WHERE):
        carried = under[1:]
    elif tree.head in (ops.POW, ops.ISEL):
        carried = under[:1]
    else:
        carried = under
    ports = frozenset().union(*(ports for ports, _ in carried))
    constants = frozenset().union(*(constants for _, constants in carried))
    plain = ports - dimensionalities.keys()
    declared = ports & dimensionalities.keys()
    if plain and declared:
        msg = (
            "mixes declared and plain ports in one operation: plain "
            f"{', '.join(map(repr, sorted(plain)))} with declared {', '.join(map(repr, sorted(declared)))}. "
            "Declare the plain ports, or keep them out of the declared ports' arithmetic."
        )
        raise BiltEquationDimensionalityError(equation_name, msg)
    if plain and constants:
        msg = (
            "mixes a plain port with a quantity constant in one operation: plain "
            f"{', '.join(map(repr, sorted(plain)))} with a constant of {', '.join(sorted(map(str, constants)))}. "
            "Declare the port, or write the constant as a bare number."
        )
        raise BiltEquationDimensionalityError(equation_name, msg)
    return ports, constants

__all__ = [
    "BasicEquation",
    "Equation",
    "equation",
]

import inspect
import types
import typing

from bilt.core._errors import BiltEquationReturnError, BiltEquationSignatureError, BiltUnresolvedAnnotationError
from bilt.core._interface import Interface
from bilt.core._port import QualifiedPort
from bilt.core._port_declaration import port_declarations
from bilt.core._sub_system import validate_name
from bilt.core._tensors._tensors import Tensor
from bilt.symbolic._expressions import Expr, SubSystemExpr, symbol_expr, tensor_symbol_expr
from bilt.symbolic._tree import Tree, symbol_tree
from bilt.symbolic._walk import subtrees

if typing.TYPE_CHECKING:
    import collections.abc

    from bilt.core._port_declaration import PortDeclaration


class Equation[R, **P](typing.Protocol):
    __bilt_name__: str
    __bilt_parameters__: tuple[str, ...]
    __bilt_port_declarations__: dict[str, PortDeclaration]
    __bilt_equation_symbols__: dict[str, Expr]
    __bilt_equation_tree__: Tree
    __bilt_sub_systems__: types.MappingProxyType[str, Interface]

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R: ...


class BasicEquation[R, **P]:
    __bilt_name__: str
    __bilt_parameters__: tuple[str, ...]
    __bilt_port_declarations__: dict[str, PortDeclaration]
    __bilt_equation_symbols__: dict[str, Expr]
    __bilt_equation_tree__: Tree
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
        symbols = _build_symbols(signature, sub_systems)
        tree = _build_equation_tree(name, function, symbols)
        equation_symbols, referenced_sub_systems, sub_system_declarations = _expand_sub_system_parameters(
            symbols, sub_systems, tree
        )

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
    """Bind each parameter to what the equation function sees when called symbolically."""
    symbols: dict[str, Expr | SubSystemExpr] = {}

    for name, parameter in signature.parameters.items():
        if name in sub_systems:
            symbols[name] = SubSystemExpr(
                name,
                {port: QualifiedPort(name, port).dotted for port in sub_systems[name].__bilt_parameters__},
            )
        elif parameter.annotation is Tensor:
            symbols[name] = tensor_symbol_expr(name)
        else:
            symbols[name] = symbol_expr(name)

    return symbols


def _expand_sub_system_parameters(
    symbols: dict[str, Expr | SubSystemExpr],
    sub_systems: collections.abc.Mapping[str, Interface],
    tree: Tree,
) -> tuple[dict[str, Expr], dict[str, Interface], dict[str, PortDeclaration]]:
    """Replace each sub-system parameter with the dotted ports the equation actually references.

    Keys keep signature order, so the parameter tuple reads straight off the result. A port the
    equation never mentions is not a parameter of it, so a sub-system contributes only what it uses.
    Each dotted port carries the declaration its sub-system's interface gives it.
    """
    occurring = subtrees(tree)
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
                sub_system_declarations[dotted] = sub_systems[name].__bilt_port_declarations__[port]

    return equation_symbols, referenced_sub_systems, sub_system_declarations


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

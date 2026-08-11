__all__ = [
    "BasicEquation",
    "Equation",
    "equation",
]

import inspect
import types
import typing

from bilt.core._error import EquationReturnError
from bilt.core._interface import Interface
from bilt.core._port import QualifiedPort
from bilt.symbolic._expressions import Expr, SubSystemExpr, symbol_expr
from bilt.symbolic._tree import Tree, symbol_tree
from bilt.symbolic._walk import subtrees

if typing.TYPE_CHECKING:
    import collections.abc


class Equation[R, **P](typing.Protocol):
    __bilt_name__: str
    __bilt_parameters__: tuple[str, ...]
    __bilt_equation_symbols__: dict[str, Expr]
    __bilt_equation_tree__: Tree
    __bilt_sub_systems__: types.MappingProxyType[str, Interface]

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R: ...


class BasicEquation[R, **P]:
    __bilt_name__: str
    __bilt_parameters__: tuple[str, ...]
    __bilt_equation_symbols__: dict[str, Expr]
    __bilt_equation_tree__: Tree
    __bilt_sub_systems__: types.MappingProxyType[str, Interface]
    _function: collections.abc.Callable[P, R]

    def __init__(self, function: collections.abc.Callable[P, R]) -> None:
        name = _function_name(function)
        signature = inspect.signature(function)

        _validate_equation_signature(name, signature)

        symbols: dict[str, Expr | SubSystemExpr] = {}
        sub_systems: dict[str, Interface] = {}

        for param_name, param in signature.parameters.items():
            annotation = param.annotation
            if isinstance(annotation, Interface):
                sub_systems[param_name] = annotation
                symbols[param_name] = SubSystemExpr(
                    param_name,
                    {port: QualifiedPort(param_name, port).dotted for port in annotation.__bilt_parameters__},
                )
            else:
                symbols[param_name] = symbol_expr(param_name)

        tree = _build_equation_tree(name, function, symbols)
        equation_symbols = _expand_sub_system_parameters(symbols, sub_systems, tree)

        self.__bilt_name__ = name
        self.__bilt_parameters__ = tuple(equation_symbols)
        self.__bilt_equation_symbols__ = equation_symbols
        self.__bilt_equation_tree__ = tree
        self.__bilt_sub_systems__ = types.MappingProxyType(sub_systems)
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
        if param.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            msg = (
                f"Equation function {name!r} must use only named parameters "
                "(no positional-only parameters, *args, or **kwargs)."
            )
            raise TypeError(msg)


def _expand_sub_system_parameters(
    symbols: dict[str, Expr | SubSystemExpr],
    sub_systems: dict[str, Interface],
    tree: Tree,
) -> dict[str, Expr]:
    """Expand each sub-system parameter into the dotted ports the equation actually uses.

    Keys keep signature order, so the caller can read the parameter tuple straight off the result.
    """
    occurring = subtrees(tree)
    equation_symbols: dict[str, Expr] = {}

    for param_name, symbol in symbols.items():
        interface = sub_systems.get(param_name)
        if interface is None:
            equation_symbols[param_name] = typing.cast("Expr", symbol)
            continue

        used_ports = sorted(
            port
            for port in interface.__bilt_parameters__
            if symbol_tree(QualifiedPort(param_name, port).dotted) in occurring
        )

        for port in used_ports:
            qualified_name = QualifiedPort(param_name, port).dotted
            equation_symbols[qualified_name] = symbol_expr(qualified_name)

    return equation_symbols


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
        raise EquationReturnError(_equation_return_error_message(name, result))

    if not result.is_relation:
        raise EquationReturnError(_equation_return_error_message(name, result))

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

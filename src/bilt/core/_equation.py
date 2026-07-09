__all__ = [
    "BasicEquation",
    "Equation",
    "equation",
]

import inspect
import typing

from bilt.core._error import EquationReturnError
from bilt.symbolic._expressions import Expr, symbol_expr
from bilt.symbolic._tree import Tree

if typing.TYPE_CHECKING:
    import collections.abc


@typing.runtime_checkable
class Equation[R, **P](typing.Protocol):
    __bilt_name__: str
    __bilt_parameters__: tuple[str, ...]
    __bilt_equation_symbols__: dict[str, Expr]
    __bilt_equation_tree__: Tree

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R: ...


class BasicEquation[R, **P]:
    __bilt_name__: str
    __bilt_parameters__: tuple[str, ...]
    __bilt_equation_symbols__: dict[str, Expr]
    __bilt_equation_tree__: Tree
    _function: collections.abc.Callable[P, R]

    def __init__(self, function: collections.abc.Callable[P, R]) -> None:
        name = _function_name(function)
        signature = inspect.signature(function)

        _validate_equation_signature(name, signature)

        parameters = tuple(signature.parameters.keys())
        symbols = _build_symbols(parameters)
        tree = _build_equation_tree(name, function, symbols)

        self.__bilt_name__ = name
        self.__bilt_parameters__ = parameters
        self.__bilt_equation_symbols__ = symbols
        self.__bilt_equation_tree__ = tree
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


def _build_symbols(parameters: tuple[str, ...]) -> dict[str, Expr]:
    return {name: symbol_expr(name) for name in parameters}


def _build_equation_tree[R, **P](
    name: str,
    function: collections.abc.Callable[P, R],
    symbols: dict[str, Expr],
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

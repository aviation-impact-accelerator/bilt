__all__ = [
    "equation",
]
import inspect
import typing

from bilt.core._error import EquationReturnError
from bilt.symbolic._expressions import Expr
from bilt.symbolic._relations import Relation, as_relation

if typing.TYPE_CHECKING:
    import collections.abc


class Equation[R, **P](typing.Protocol):
    __bilt_equation_name__: str
    __bilt_equation_parameters__: tuple[str, ...]
    __bilt_equation_symbols__: dict[str, Expr]
    __bilt_equation_relation__: Relation

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R: ...


class BasicEquation[R, **P]:
    __bilt_equation_name__: str
    __bilt_equation_parameters__: tuple[str, ...]
    __bilt_equation_symbols__: dict[str, Expr]
    __bilt_equation_relation__: Relation
    _function: collections.abc.Callable[P, R]

    def __init__(self, function: collections.abc.Callable[P, R]) -> None:
        name = _function_name(function)
        signature = inspect.signature(function)

        _validate_equation_signature(name, signature)

        parameters = tuple(signature.parameters.keys())
        symbols = _build_symbols(parameters)
        relation = _build_equation_relation(name, function, symbols)

        self.__bilt_equation_name__ = name
        self.__bilt_equation_parameters__ = parameters
        self.__bilt_equation_symbols__ = symbols
        self.__bilt_equation_relation__ = relation
        self._function = function

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        return self._function(*args, **kwargs)

    def __str__(self) -> str:
        return f"{self.__bilt_equation_name__}: {self.__bilt_equation_relation__}"


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
    return {name: Expr.symbol(name) for name in parameters}


def _build_equation_relation[R, **P](
    name: str,
    function: collections.abc.Callable[P, R],
    symbols: dict[str, Expr],
) -> Relation:
    symbolic_function = typing.cast(
        "collections.abc.Callable[..., object]",
        function,
    )
    result = symbolic_function(**symbols)

    try:
        return as_relation(result)
    except TypeError:
        raise EquationReturnError(_equation_return_error_message(name, result)) from None


def _equation_return_error_message(name: str, result: object) -> str:
    if isinstance(result, Expr):
        return (
            f"Equation {name!r} returned an expression, "
            "not an equation.\n\n"
            "Did you forget to compare it with another expression?\n\n"
            "Use:\n"
            "    return equals(lhs, rhs)\n"
            "or:\n"
            "    return lhs == rhs"
        )

    return (
        f"Equation {name!r} must return an equation relation, "
        f"but returned {type(result).__name__!r}.\n\n"
        "Use:\n"
        "    return equals(lhs, rhs)\n"
        "or:\n"
        "    return lhs == rhs"
    )


def equation[R, **P](
    function: collections.abc.Callable[P, R],
) -> BasicEquation[R, P]:
    """Decorator to denote a function as an equation."""
    return BasicEquation(function)

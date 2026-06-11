__all__ = [
    "equation",
]

import collections.abc
import inspect
import typing

from bilt.core._error import EquationReturnError
from bilt.symbolic._expressions import Expr
from bilt.symbolic._relations import Relation

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
        self.__bilt_equation_name__ = getattr(
            function,
            "__name__",
            type(function).__name__,
        )

        signature = inspect.signature(function)
        for param in signature.parameters.values():
            if param.kind in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                msg = (
                    f"Equation function {self.__bilt_equation_name__!r} must use only named parameters "
                    "(no positional-only parameters, *args, or **kwargs)."
                )
                raise TypeError(msg)

        self.__bilt_equation_parameters__ = tuple(signature.parameters.keys())

        self.__bilt_equation_symbols__ = {name: Expr.symbol(name) for name in self.__bilt_equation_parameters__}

        symbolic_function = typing.cast(
            "collections.abc.Callable[..., object]",
            function,
        )
        relation = symbolic_function(**self.__bilt_equation_symbols__)

        if not isinstance(relation, Relation):
            if isinstance(relation, Expr):
                msg = (
                    f"Equation {self.__bilt_equation_name__!r} returned an expression, "
                    "not an equation.\n\n"
                    "Did you forget to compare it with another expression?\n\n"
                    "Use:\n"
                    "    return equals(lhs, rhs)\n"
                    "or:\n"
                    "    return lhs == rhs"
                )
            else:
                msg = (
                    f"Equation {self.__bilt_equation_name__!r} must return an equation relation, "
                    f"but returned {type(relation).__name__!r}.\n\n"
                    "Use:\n"
                    "    return equals(lhs, rhs)\n"
                    "or:\n"
                    "    return lhs == rhs"
                )

            raise EquationReturnError(msg)

        self.__bilt_equation_relation__ = relation
        self._function = function

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        return self._function(*args, **kwargs)

    def __str__(self) -> str:
        return f"{self.__bilt_equation_name__}: {self.__bilt_equation_relation__}"


def equation[R, **P](
    function: collections.abc.Callable[P, R],
) -> BasicEquation[R, P]:
    """Decorator to denote a function as an equation."""
    return BasicEquation(function)

__all__ = ["transform"]

import inspect
import typing

from bilt.core._errors import BiltTransformDimensionalityError
from bilt.core._port_declaration import port_declarations
from bilt.units import Quantity
from bilt.units._errors import BiltDimensionalityError
from bilt.units._units import BASE_UNIT_SYSTEM

if typing.TYPE_CHECKING:
    import collections.abc


class Transform[R, **P](typing.Protocol):
    __bilt_name__: str
    __bilt_parameters__: tuple[str, ...]

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R: ...


class BasicTransform[R, **P]:
    def __init__(self, function: collections.abc.Callable[P, R]) -> None:
        self.__bilt_name__ = function.__name__  # ty: ignore[unresolved-attribute]
        signature = inspect.signature(function)
        self.__bilt_parameters__ = tuple(signature.parameters.keys())
        self.__bilt_port_declarations__ = port_declarations(
            self.__bilt_name__,
            {port: parameter.annotation for port, parameter in signature.parameters.items()}
            | {self.__bilt_name__: signature.return_annotation},
        )
        self._function = function
        _probe(self)

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        return self._function(*args, **kwargs)


def transform[R, **P](function: collections.abc.Callable[P, R]) -> BasicTransform[R, P]:
    """Decorator to denote a function as a transform."""
    return BasicTransform(function)


def _probe(transform: BasicTransform[object, ...]) -> None:
    """Run a transform whose every parameter declares a dimensionality on magnitude-1 quantities of each."""
    dimensionalities = {
        parameter: transform.__bilt_port_declarations__[parameter].dimensionality
        for parameter in transform.__bilt_parameters__
    }
    probes = {
        parameter: Quantity(1.0, BASE_UNIT_SYSTEM.coherent_unit(dimensionality))
        for parameter, dimensionality in dimensionalities.items()
        if dimensionality is not None
    }
    if len(probes) != len(dimensionalities):
        return

    try:
        result = transform(**probes)
    except BiltDimensionalityError as error:
        raise BiltTransformDimensionalityError(
            transform.__bilt_name__, f"is dimensionally inconsistent: {error}"
        ) from error
    except Exception:  # noqa: BLE001 -- the body cannot run on quantities, so its annotation stands
        return

    declared = transform.__bilt_port_declarations__[transform.__bilt_name__].dimensionality
    if declared is not None and isinstance(result, Quantity) and result.unit.dimension is not declared:
        msg = f"returns {result.unit.dimension} when probed, but its return annotation declares {declared}"
        raise BiltTransformDimensionalityError(transform.__bilt_name__, msg)

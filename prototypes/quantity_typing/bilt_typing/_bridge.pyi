from collections.abc import Callable, Iterable
from typing import Generic, TypeVar
from bilt import Dim, Interface
from bilt import Implementation as NativeImplementation
from bilt.core._equation import BasicEquation
from bilt.core._transform import BasicTransform
from bilt.core._model_constituent import BiltModelConstituent
from ._markers import DimensionType
from ._values import Quantity, Tensor

class Requires:
    axis: Dim
    def __init__(self, axis: Dim) -> None: ...

def system[T](cls: type[T]) -> type[T]: ...
def equation[**P, R](function: Callable[P, R]) -> BasicEquation[R, P]: ...

class Implementation(NativeImplementation):
    def __init__(self, system: Interface, components: Iterable[BiltModelConstituent]) -> None: ...

# A concrete call compares values; symbolic tracing builds a relation tree.
# The same quantity dimension is enforced statically without changing __eq__.
def equal[D: DimensionType, L: float | Tensor[float], R: float | Tensor[float]](
    left: Quantity[D, L],
    right: Quantity[D, R],
) -> bool: ...

D = TypeVar("D", bound=DimensionType)

class Pointwise(Generic[D]):
    def __call__[R: float | Tensor[float]](self, value: Quantity[D, R]) -> Quantity[D, R]: ...

def pointwise[D: DimensionType](function: Callable[[Quantity[D, float]], Quantity[D, float]]) -> Pointwise[D]: ...
def transform[**P, R](function: Callable[P, R]) -> BasicTransform[R, P]: ...

import dataclasses
import typing

from bilt.core._tensors._tensor_analysis import DomainedVar
from bilt.symbolic._domains import Domain

if typing.TYPE_CHECKING:
    import collections.abc


@dataclasses.dataclass
class BackendCallable:
    name: str
    parameters: dict[str, DomainedVar]
    return_domain: Domain
    union_domain: Domain
    callable: CallableType

type CallableType = collections.abc.Callable[..., typing.Any]
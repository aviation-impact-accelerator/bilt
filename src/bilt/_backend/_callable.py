import dataclasses
import typing

if typing.TYPE_CHECKING:
    from bilt.symbolic._domains import CanonicalisedDomain
    
    import collections.abc


@dataclasses.dataclass
class BackendCallable:
    name: str
    expected_parameters: dict[str, CanonicalisedDomain]
    expected_return_domain: CanonicalisedDomain
    expected_union_domain: CanonicalisedDomain
    callable: CallableType

type CallableType = collections.abc.Callable[..., typing.Any]
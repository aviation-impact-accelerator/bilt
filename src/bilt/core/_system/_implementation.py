__all__ = ["Implementation"]

import types
import typing

from bilt.core._system._analysis import (
    build_backend_case,
    find_successful_case,
    validate_broadcasts,
)
from bilt.core._system._evaluation import domain_registry_scope
from bilt.core._tensor._tensor import BiltTensor

if typing.TYPE_CHECKING:
    import collections.abc

    from bilt._backend import Backend, BackendCallable
    from bilt.core._model_constituent import BiltModelConstituent
    from bilt.core._system._domain_registry import DomainRegistry

    from ._interface import Interface


type InputDomainSignature = tuple[
    tuple[str, tuple[object, ...]],
    ...,
]

type CallableCacheKey = tuple[
    str,
    InputDomainSignature,
    Backend,
]


class Implementation:
    """Register a collection of transforms as an implementation of a system."""

    def __init__(
        self,
        system: Interface,
        components: collections.abc.Iterable[BiltModelConstituent],
    ) -> None:
        self._system = system
        self._components: types.MappingProxyType[
            str,
            BiltModelConstituent,
        ] = types.MappingProxyType({component.__bilt_name__: component for component in components})
        self._callable_cache: dict[
            CallableCacheKey,
            BackendCallable,
        ] = {}

        system.__bilt_system_registry__.add(self)

    @property
    def system(self) -> Interface:
        return self._system

    def evaluate(
        self,
        inputs: dict[str, object],
        output: str,
        backend: Backend,
    ) -> object:
        with domain_registry_scope() as registry:
            for value in inputs.values():
                _observe_tensor(value, registry)

            if output in inputs:
                return inputs[output]

            input_signature = _input_domain_signature(inputs)
            cache_key = (
                output,
                input_signature,
                backend,
            )

            backend_callable = self._callable_cache.get(cache_key)

            if backend_callable is None:
                transform_dag = find_successful_case(
                    self,
                    inputs,
                    output,
                    backend,
                )

                validated_dag = validate_broadcasts(
                    case=transform_dag,
                    inputs=inputs,
                    registry=registry,
                )

                backend_case = build_backend_case(
                    validated_dag,
                    backend,
                )
                backend_callable = backend.chain_case(backend_case)
                self._callable_cache[cache_key] = backend_callable

            result = backend_callable.callable(**inputs)
            _observe_tensor(result, registry)

            return result

    def __enter__(self) -> typing.Self:
        self._system.__bilt_system_implementation_default__.append(self)
        return self

    def __exit__(self, *_: object) -> None:
        self._system.__bilt_system_implementation_default__.pop()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Implementation):
            return NotImplemented

        return self.system == other.system and self._components == other._components

    def __hash__(self) -> int:
        return hash(
            (
                self.system,
                frozenset(self._components.items()),
            )
        )

    def __repr__(self) -> str:
        components = frozenset(self._components.values())

        return f"{type(self).__name__}(system={self.system}, transforms={components!r})"


def _input_domain_signature(
    inputs: collections.abc.Mapping[str, object],
) -> InputDomainSignature:
    """Return input names and their ordered symbolic dimensions."""
    return tuple(
        (
            name,
            _tensor_domain_dimensions(value),
        )
        for name, value in inputs.items()
    )


def _tensor_domain_dimensions(
    value: object,
) -> tuple[object, ...]:
    if not isinstance(value, BiltTensor):
        return ()

    return tuple(value.domain.dims)


def _observe_tensor(
    value: object,
    registry: DomainRegistry,
) -> None:
    if isinstance(value, BiltTensor):
        value.observe(registry)

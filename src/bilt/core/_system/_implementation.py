__all__ = ["Implementation"]

import types
import typing

import numpy as np

from bilt._backend._python._backend import PythonBackend
from bilt.core._system._analysis import (
    build_backend_case,
    build_broadcast_plan,
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

            broadcast_plan = build_broadcast_plan(inputs, registry)

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
                    broadcast_plan,
                )

                backend_callable = backend.chain_case(backend_case)
                self._callable_cache[cache_key] = backend_callable

            # Note that the Python backend is still relying on the BiltTensor operator overloads,
            # so for now we forego the conversion to raw arrays to keep the current Python backend
            # working.

            if not isinstance(backend, PythonBackend):
                inputs = self._tensor_inputs_to_raw_array_type(inputs)

            result = typing.cast("object", backend_callable.callable(**inputs))  # pyright: ignore[reportUnknownMemberType]

            if not broadcast_plan.is_scalar_only and not isinstance(backend, PythonBackend):
                # If there are any tensor inputs (i.e. this isn't a strictly scalar calculation) we will have a tensor output
                result = BiltTensor(np.asarray(result), broadcast_plan.output_domain)
                _observe_tensor(result, registry)

            return result

    def _tensor_inputs_to_raw_array_type(self, inputs: dict[str, object]) -> dict[str, object]:
        return {name: (obj.values if isinstance(obj, BiltTensor) else obj) for name, obj in inputs.items()}

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

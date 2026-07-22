from __future__ import annotations

__all__ = ["Implementation"]

import types
import typing

from bilt.core._system._analysis import find_successful_case
from bilt.core._system._evaluation import domain_registry_scope

if typing.TYPE_CHECKING:
    import collections.abc

    from bilt._backend import Backend, BackendCallable
    from bilt.core._model_constituent import BiltModelConstituent

    from ._interface import Interface


class Implementation:
    """Register a collection of transforms as an implementation of a system."""

    def __init__(
        self,
        system: Interface,
        components: collections.abc.Iterable[BiltModelConstituent],
    ) -> None:
        self._system: Interface = system
        self._components: types.MappingProxyType[
            str,
            BiltModelConstituent,
        ] = types.MappingProxyType(
            {
                component.__bilt_name__: component
                for component in components
            }
        )
        self._callable_cache: dict[
            tuple[str, tuple[str, ...], Backend],
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
        """Evaluate an output in a private, evaluation-scoped registry."""

        with domain_registry_scope():
            if output in inputs:
                return inputs[output]

            key = (output, tuple(inputs), backend)
            backend_callable = self._callable_cache.get(key)

            if backend_callable is None:
                backend_case = find_successful_case(
                    implementation=self,
                    inputs=inputs,
                    output=output,
                    backend=backend,
                )
                backend_callable = backend.chain_case(backend_case)
                self._callable_cache[key] = backend_callable

            return backend_callable.callable(**inputs)

    def __enter__(self) -> typing.Self:
        self._system.__bilt_system_implementation_default__.append(self)
        return self

    def __exit__(self, *_: object) -> None:
        self._system.__bilt_system_implementation_default__.pop()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Implementation):
            return NotImplemented

        return (
            self.system == other.system
            and self._components == other._components
        )

    def __hash__(self) -> int:
        return hash(
            (
                self.system,
                frozenset(self._components.items()),
            )
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"system={self.system}, "
            f"transforms={frozenset(self._components.values())!r})"
        )

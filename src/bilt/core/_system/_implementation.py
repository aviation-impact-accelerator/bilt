__all__ = ["Implementation"]

import dataclasses
import types
import typing

from bilt.core._structural_analysis._protocol import (
    get_current_structural_analysis_backend,
)

if typing.TYPE_CHECKING:
    import collections.abc

    from bilt._backend import Backend, BackendCallable, BackendCase
    from bilt.core._interface import Interface
    from bilt.core._model_constituent import BiltModelConstituent


@dataclasses.dataclass(frozen=True, slots=True)
class _EvaluationRequest:
    """A request to compute an output from a set of available values."""

    output: str
    available: frozenset[str]


@dataclasses.dataclass(frozen=True, slots=True)
class _Executable:
    """A backend callable ready for evaluation."""

    backend_callable: BackendCallable

    def evaluate(
        self,
        inputs: collections.abc.Mapping[str, object],
    ) -> object:
        return self.backend_callable.callable(**{name: inputs[name] for name in self.backend_callable.parameters})


class Implementation:
    """Register a collection of components as an implementation of a system."""

    def __init__(
        self,
        system: Interface,
        components: collections.abc.Iterable[BiltModelConstituent],
    ) -> None:
        self._system = system
        self._components = types.MappingProxyType({component.__bilt_name__: component for component in components})
        self._executable_cache: dict[
            tuple[_EvaluationRequest, Backend, object],
            _Executable,
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
        if output in inputs:
            return inputs[output]

        request = _EvaluationRequest(
            output=output,
            available=frozenset(inputs),
        )

        executable = self._get_executable(request, backend)

        return executable.evaluate(inputs)

    def _get_executable(
        self,
        request: _EvaluationRequest,
        backend: Backend,
    ) -> _Executable:
        structural_analysis = get_current_structural_analysis_backend()

        cache_key = (
            request,
            backend,
            structural_analysis,
        )

        try:
            return self._executable_cache[cache_key]
        except KeyError:
            pass

        planned_case = self._plan(
            request,
            backend,
            structural_analysis,
        )

        executable = self._compile(
            planned_case,
            backend,
        )

        self._executable_cache[cache_key] = executable

        return executable

    def _plan(
        self,
        request: _EvaluationRequest,
        backend: Backend,
        structural_analysis: typing.Callable[..., BackendCase],
    ) -> BackendCase:
        return structural_analysis(
            tuple(self._components.values()),
            set(request.available),
            request.output,
            backend,
        )

    @staticmethod
    def _compile(
        planned_case: BackendCase,
        backend: Backend,
    ) -> _Executable:
        return _Executable(backend_callable=backend.chain_case(planned_case))

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
        return f"{self.__class__.__name__}(system={self.system}, components={frozenset(self._components.values())!r})"

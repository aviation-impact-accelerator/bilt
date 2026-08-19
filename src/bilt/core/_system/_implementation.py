from bilt.core._system._lowering_and_lifting import generate_lifting_plan, lift_output, lower_inputs
from bilt.core._tensors._tensor_alignment import generate_alignment_plan

__all__ = ["Implementation"]

import dataclasses
import types
import typing

from bilt.core._evaluation_context import EvaluationContext
from bilt.core._structural_analysis._protocol import (
    get_current_structural_analysis_backend,
)

if typing.TYPE_CHECKING:
    import collections.abc

    from bilt._backend import Backend, BackendCallable
    from bilt.core._interface import Interface
    from bilt.core._model_constituent import BiltModelConstituent


@dataclasses.dataclass(frozen=True, slots=True)
class _EvaluationRequest:
    """A request to compute an output from a set of available values."""

    output: str
    available: frozenset[str]


class Implementation:
    """Register a collection of transforms as an implementation of a system."""

    def __init__(
        self,
        system: Interface,
        components: collections.abc.Iterable[BiltModelConstituent],
    ) -> None:
        self._system: Interface = system
        self._components: types.MappingProxyType[str, BiltModelConstituent] = types.MappingProxyType(
            {component.__bilt_name__: component for component in components}
        )
        self._evaluation_cache: dict[
            tuple[_EvaluationRequest, Backend],
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
        with EvaluationContext() as evaluation:
            evaluation.domains.bind_all(inputs.values())

            if output in inputs:
                output_value = inputs[output]
            else:
                request = _EvaluationRequest(
                    output=output,
                    available=frozenset(inputs),
                )

                backend_callable = self._get_evaluation(request, backend)

                lowered_inputs = lower_inputs(inputs)
                alignment_plan = generate_alignment_plan(inputs)
                aligned_inputs = backend.apply_alignment(lowered_inputs, alignment_plan)
                raw_output_value = backend_callable.evaluate(aligned_inputs)

                lifting_plan = generate_lifting_plan(inputs)
                output_value = lift_output(raw_output_value, lifting_plan)

            evaluation.domains.bind(output_value)

        return output_value

    def _get_evaluation(
        self,
        request: _EvaluationRequest,
        backend: Backend,
    ) -> BackendCallable:
        cache_key = (
            request,
            backend,
        )

        if cached := self._evaluation_cache.get(cache_key):
            return cached

        structural_analysis_backend = get_current_structural_analysis_backend()

        planned_case = structural_analysis_backend(
            tuple(self._components.values()),
            set(request.available),
            request.output,
            backend,
        )

        backend_callable = backend.chain_case(planned_case)
        self._evaluation_cache[cache_key] = backend_callable

        return backend_callable

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
        return f"{self.__class__.__name__}(system={self.system}, transforms={frozenset(self._components.values())!r})"

from bilt.core._system._lowering_and_lifting import generate_lifting_plan, lift_output, lower_inputs
from bilt.core._tensors._tensor_alignment import generate_alignment_plan

__all__ = ["Implementation"]

import dataclasses
import types
import typing

from bilt.core._equation import BasicEquation
from bilt.core._evaluation_context import EvaluationContext, current_evaluation
from bilt.core._evaluation_request import EvaluationRequest
from bilt.core._port import mangle_inputs
from bilt.core._structural_analysis._causalization_block_solver._structural_scope import StructuralAnalysis
from bilt.core._structural_analysis._exhaustive import ExhaustiveStructuralAnalyser
from bilt.core._sub_system import SubSystemConstituent
from bilt.core._system._dimensional_unification import check_dimensional_consistency
from bilt.core._system._errors import (
    BiltSubSystemConflictError,
    BiltSubSystemMismatchedImplementationError,
    BiltSubSystemMissingImplementationError,
    BiltSubSystemNameCollisionError,
    BiltSubSystemUnreferencedEntryError,
)
from bilt.core._system._input_validation import validate_inputs
from bilt.core._tensors._tensors import Tensor
from bilt.symbolic._domains import Domain

if typing.TYPE_CHECKING:
    import collections.abc

    from bilt._backend import Backend, BackendCallable
    from bilt.core._interface import Interface
    from bilt.core._model_constituent import BiltModelConstituent
    from bilt.core._port_declaration import PortDeclaration
    from bilt.core._structural_analysis import StructuralAnalyser
    from bilt.symbolic._dimensions import Coordinate, Dim


@dataclasses.dataclass(frozen=True)
class _CompilationRequest:
    output: str
    input_domains: frozenset[tuple[str, Domain]]
    backend: Backend
    expose_intermediates: bool


@dataclasses.dataclass(frozen=True)
class _CompilationContext:
    canonical_order: Domain
    coordinate_signature: tuple[
        tuple[Dim, tuple[Coordinate, ...]],
        ...,
    ]


class Implementation:
    """Register a collection of transforms as an implementation of a system."""

    def __init__(
        self,
        system: Interface,
        components: collections.abc.Iterable[BiltModelConstituent],
        *,
        internal_implementations: collections.abc.Mapping[str, Implementation] | None = None,
    ) -> None:
        self._system: Interface = system
        named_components = {component.__bilt_name__: component for component in components}

        internal = internal_implementations or {}

        sub_systems = _build_sub_system_constituents(named_components, internal)
        _validate_sub_systems(named_components, internal, sub_systems)
        self._components: types.MappingProxyType[str, BiltModelConstituent] = types.MappingProxyType(
            named_components | sub_systems
        )
        check_dimensional_consistency(system, self._components.values())
        self._port_declarations: dict[str, PortDeclaration] = {
            port: declaration
            for component in self._components.values()
            for port, declaration in component.__bilt_port_declarations__.items()
        } | system.__bilt_port_declarations__
        self._evaluation_cache: dict[
            _CompilationRequest,
            dict[_CompilationContext, BackendCallable],
        ] = {}
        self._structural_analysis: StructuralAnalysis | None = None
        self._analysers: dict[Backend, StructuralAnalyser] = {}
        system.__bilt_system_registry__.add(self)

    @property
    def system(self) -> Interface:
        return self._system

    def structural_analysis(self) -> StructuralAnalysis:
        """Return backend-independent analysis of this implementation's structure."""
        if self._structural_analysis is None:
            self._structural_analysis = StructuralAnalysis(tuple(self._components.values()))
        return self._structural_analysis

    def analyser_for(self, backend: Backend) -> StructuralAnalyser:
        """The structural analyser for this implementation's components under ``backend``."""
        analyser = self._analysers.get(backend)
        if analyser is None:
            analyser = ExhaustiveStructuralAnalyser(tuple(self._components.values()), backend)
            self._analysers[backend] = analyser
        return analyser

    def evaluate(
        self,
        inputs: dict[str, object],
        output: str,
        backend: Backend,
    ) -> object:
        """Calculate the requested value without retaining intermediate results."""
        return self._evaluate(inputs, output, backend, expose_intermediates=False)[output]

    def _evaluate_with_intermediates(
        self,
        inputs: dict[str, object],
        output: str,
        backend: Backend,
    ) -> dict[str, object]:
        """Calculate and lift every port computed in this implementation's plan."""
        return self._evaluate(inputs, output, backend, expose_intermediates=True)

    def _evaluate(
        self,
        inputs: dict[str, object],
        output: str,
        backend: Backend,
        *,
        expose_intermediates: bool,
    ) -> dict[str, object]:
        validate_inputs(self._system.__bilt_name__, self._port_declarations, inputs)

        with EvaluationContext() as evaluation:
            evaluation.domains.bind_all(inputs.values())

            if output in inputs:
                outputs = {output: inputs[output]}
            else:
                request = EvaluationRequest(
                    output=output,
                    input_domains={
                        name: value.domain if isinstance(value, Tensor) else Domain.scalar()
                        for name, value in inputs.items()
                    },
                )

                backend_callable = self.get_evaluation(request, backend, expose_intermediates=expose_intermediates)

                lowered_inputs = lower_inputs(inputs)
                alignment_plan = generate_alignment_plan(inputs)
                aligned_inputs = backend.apply_alignment(lowered_inputs, alignment_plan)
                backend_inputs = mangle_inputs(aligned_inputs)
                if expose_intermediates:
                    raw_outputs = backend_callable.evaluate_outputs(backend_inputs)
                    output_domains = {result.port: result.domain for result in backend_callable.outputs or ()}
                else:
                    raw_outputs = {output: backend_callable.evaluate(backend_inputs)}
                    output_domains = {output: typing.cast("Domain", backend_callable.output_domain)}

                outputs = {
                    name: lift_output(
                        value,
                        generate_lifting_plan(output_domains[name], self._port_declarations.get(name)),
                    )
                    for name, value in raw_outputs.items()
                }

            validate_inputs(self._system.__bilt_name__, self._port_declarations, outputs)
            evaluation.domains.bind_all(outputs.values())

        return outputs

    def get_evaluation(
        self,
        request: EvaluationRequest,
        backend: Backend,
        *,
        expose_intermediates: bool = False,
    ) -> BackendCallable:
        evaluation = current_evaluation()

        compilation_request = _CompilationRequest(
            output=request.output,
            input_domains=frozenset(request.input_domains.items()),
            backend=backend,
            expose_intermediates=expose_intermediates,
        )
        compilation_context = _CompilationContext(
            canonical_order=evaluation.canonical_order,
            coordinate_signature=evaluation.domains.coordinate_signature(),
        )

        cached_evaluations = self._evaluation_cache.get(compilation_request)
        if cached_evaluations is not None:
            cached = cached_evaluations.get(compilation_context)
            if cached is not None:
                return cached

        planned_case = self.analyser_for(backend).analyse(request)

        backend_callable = backend.chain_case(
            dataclasses.replace(planned_case, expose_intermediates=expose_intermediates)
        )
        if planned_case.intermediates:
            output_domain = planned_case.intermediates[-1].output_domain
        else:
            output_domain = request.input_domains[request.output]

        backend_callable.output_domain = output_domain

        self._evaluation_cache.setdefault(
            compilation_request,
            {},
        )[compilation_context] = backend_callable

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


def _required_sub_system_interfaces(
    components: collections.abc.Iterable[BiltModelConstituent],
) -> dict[str, Interface]:
    """Read off the equations which interface each sub-system they have as a parameter ."""
    required: dict[str, Interface] = {}
    for component in components:
        if isinstance(component, BasicEquation):
            for name, interface in component.__bilt_sub_systems__.items():
                declared = required.setdefault(name, interface)
                if declared is not interface:
                    raise BiltSubSystemConflictError(name, declared.__bilt_name__, interface.__bilt_name__)
    return required


def _build_sub_system_constituents(
    components: collections.abc.Mapping[str, BiltModelConstituent],
    internal_implementations: collections.abc.Mapping[str, Implementation],
) -> dict[str, SubSystemConstituent]:
    """Collect the sub-systems into `SubSystemConstituents` ensuring no name collisions."""
    constituents = {
        name: constituent for name, constituent in components.items() if isinstance(constituent, SubSystemConstituent)
    }
    for name, implementation in internal_implementations.items():
        if name in components:
            raise BiltSubSystemNameCollisionError(name)
        constituents[name] = SubSystemConstituent(name=name, implementation=implementation)
    return constituents


def _validate_sub_systems(
    components: collections.abc.Mapping[str, BiltModelConstituent],
    internal_implementations: collections.abc.Mapping[str, Implementation],
    sub_systems: collections.abc.Mapping[str, SubSystemConstituent],
) -> None:
    """Validate we have exactly as many internal sub-systems as we need."""
    required_interfaces = _required_sub_system_interfaces(components.values())

    for name, interface in required_interfaces.items():
        constituent = sub_systems.get(name)
        if constituent is None:
            raise BiltSubSystemMissingImplementationError(
                name, interface.__bilt_name__, sorted(internal_implementations)
            )
        if constituent.interface is not interface:
            raise BiltSubSystemMismatchedImplementationError(
                name, interface.__bilt_name__, constituent.interface.__bilt_name__
            )

    for name in internal_implementations:
        if name not in required_interfaces:
            raise BiltSubSystemUnreferencedEntryError(name, sorted(required_interfaces))

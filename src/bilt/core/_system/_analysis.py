__all__ = [
    "BiltSystemEvaluationError",
    "TransformDAG",
    "ValidatedTransform",
    "ValidatedTransformDAG",
    "build_backend_case",
    "find_successful_case",
    "validate_broadcasts",
]

import ast
import dataclasses
import inspect
import textwrap
import types
import typing

from bilt._backend import BackendCallable, BackendCase
from bilt.core._equation import BasicEquation
from bilt.core._error import BiltCoreError
from bilt.core._system._interface import Interface
from bilt.core._tensor._tensor import BiltTensor, as_bilt_tensor
from bilt.core._transform import BasicTransform
from bilt.symbolic._domains import Domain

if typing.TYPE_CHECKING:
    from bilt._backend import Backend
    from bilt.core._model_constituent import BiltModelConstituent
    from bilt.core._system._domain_registry import DomainRegistry
    from bilt.core._system._implementation import Implementation


class BiltSystemEvaluationError(BiltCoreError):
    pass


class BiltBroadcastPlanError(BiltCoreError):
    pass


@dataclasses.dataclass(frozen=True, slots=True)
class TransformDAG:
    """A backend-independent, topologically ordered transform DAG."""

    name: str
    parameters: tuple[str, ...]
    operations: tuple[BiltModelConstituent, ...]
    return_name: str


@dataclasses.dataclass(frozen=True, slots=True)
class ValidatedTransform:
    """One transform annotated with its input and output domains."""

    operation: BiltModelConstituent
    parameter_domains: tuple[Domain, ...]
    result_domain: Domain


@dataclasses.dataclass(frozen=True, slots=True)
class ValidatedTransformDAG:
    """A transform DAG whose named-domain broadcasts have been validated."""

    dag: TransformDAG
    transforms: tuple[ValidatedTransform, ...]
    value_domains: types.MappingProxyType[str, Domain]

    @property
    def result_domain(self) -> Domain:
        return self.value_domains[self.dag.return_name]


@dataclasses.dataclass(frozen=True, slots=True)
class TensorPlan:
    """The plan for a single tensor."""

    name: str
    input_to_output_dim_mapping: tuple[int, ...]


@dataclasses.dataclass(frozen=True, slots=True)
class BroadcastPlan:
    """A broadcast plan for all of our tensor inputs.

    To pass from analysis (which will know the concrete inputs) to the
    backend (which can modify the AST to align the tensors).
    """

    tensor_plans: list[TensorPlan]
    output_domain: Domain

    @property
    def is_scalar_only(self) -> bool:
        return not self.tensor_plans

    @classmethod
    def empty(cls) -> BroadcastPlan:
        return cls([], Domain.scalar())


def build_broadcast_plan(inputs: dict[str, object]) -> BroadcastPlan:
    """Return a broadcast plan containing a list of plans for each input.

    Non-tensor inputs will be omitted from the plan.
    """
    tensor_inputs = [obj for obj in inputs.values() if isinstance(obj, BiltTensor)]

    if not tensor_inputs:
        return BroadcastPlan.empty()

    output_domain = _derive_output_domain(tensor_inputs)

    tensor_plans: list[TensorPlan] = []

    for name, obj in inputs.items():
        if not isinstance(obj, BiltTensor):
            continue

        input_to_output_dim_mapping = _derive_input_to_output_dim_mapping(obj, output_domain)
        tensor_plan = TensorPlan(name, input_to_output_dim_mapping)
        tensor_plans.append(tensor_plan)

    return BroadcastPlan(tensor_plans, output_domain)


def _derive_output_domain(tensors: list[BiltTensor]) -> Domain:
    """Return the domain with all dimensions found in the given tensors.

    Shared dimensions align by semantic name. Dimensions are appended
    in order, as encountered in the list of tensors.
    """
    dimension_union = list(tensors[0].domain.dims)

    for i in range(1, len(tensors)):
        tensor = tensors[i]

        for dim in tensor.domain.dims:
            if dim not in dimension_union:
                dimension_union.append(dim)

    return Domain(tuple(dimension_union))


def _derive_input_to_output_dim_mapping(tensor: BiltTensor, output_domain: Domain) -> tuple[int, ...]:
    """Return the index of each input dimension in the output domain, for each input.

    E.g.:<br>
    * Input=[1, 2, 3] over time<br>
    * Output domain is (space, time)<br>
    * Then the output of this function would be (1,), representing that the single input
    dimension should be moved to the 1-th dimension, and a singleton dimension be added
    as the 0-th dimension, to align the input tensor to the output domain.
    """
    input_to_output_dim_mapping: list[int] = []

    for input_dim in tensor.domain.dims:
        dim_index_in_output = output_domain.dims.index(input_dim)
        input_to_output_dim_mapping.append(dim_index_in_output)

    return tuple(input_to_output_dim_mapping)


def find_successful_case(
    implementation: Implementation,
    inputs: dict[str, object],
    output: str,
    backend: Backend | None = None,
) -> TransformDAG:
    """Build the backend-independent transform DAG for an output.

    ``backend`` remains accepted for compatibility with the previous API, but
    no backend work is performed at this stage.
    """
    _ = backend

    if output not in implementation._components:  # noqa: SLF001
        message = f"No operation registered to produce output {output!r}"
        raise BiltSystemEvaluationError(message)

    return TransformDAG(
        name=output,
        parameters=tuple(inputs),
        operations=_path_through_graph(
            implementation=implementation,
            inputs=inputs,
            output=output,
        ),
        return_name=output,
    )


def validate_broadcasts(
    implementation: Implementation | None = None,
    inputs: dict[str, object] | None = None,
    output: str | None = None,
    registry: DomainRegistry | None = None,
    *,
    case: TransformDAG | None = None,
) -> ValidatedTransformDAG:
    """Validate and annotate named-domain broadcasts over a transform DAG.

    The positional arguments preserve the former API:

        validate_broadcasts(implementation, inputs, output, registry)

    New code should pass an already selected DAG:

        validate_broadcasts(case=dag, inputs=inputs, registry=registry)
    """
    if inputs is None:
        msg = "inputs must be provided."
        raise TypeError(msg)

    if registry is None:
        msg = "registry must be provided."
        raise TypeError(msg)

    if case is None:
        if implementation is None:
            msg = "implementation must be provided when case is omitted."
            raise TypeError(msg)

        if output is None:
            msg = "output must be provided when case is omitted."
            raise TypeError(msg)

        case = find_successful_case(
            implementation=implementation,
            inputs=inputs,
            output=output,
        )

    resolved = {name: _input_domain(value, registry) for name, value in inputs.items()}
    validated: list[ValidatedTransform] = []

    for operation in case.operations:
        try:
            parameter_domains = tuple(resolved[parameter] for parameter in operation.__bilt_parameters__)
        except KeyError as error:
            parameter = typing.cast("str", error.args[0])
            message = f"Transform {operation.__bilt_name__!r} depends on unresolved value {parameter!r}."
            raise BiltSystemEvaluationError(message) from None

        result_domain = _operation_result_domain(
            operation,
            parameter_domains,
        )
        resolved[operation.__bilt_name__] = result_domain
        validated.append(
            ValidatedTransform(
                operation=operation,
                parameter_domains=parameter_domains,
                result_domain=result_domain,
            )
        )

    if case.return_name not in resolved:
        message = f"The transform DAG did not resolve output {case.return_name!r}."
        raise BiltSystemEvaluationError(message)

    return ValidatedTransformDAG(
        dag=case,
        transforms=tuple(validated),
        value_domains=types.MappingProxyType(dict(resolved)),
    )


def build_backend_case(case: ValidatedTransformDAG, backend: Backend, broadcast_plan: BroadcastPlan) -> BackendCase:
    """Transpile a validated transform DAG into a backend case."""
    return BackendCase(
        name=case.dag.name,
        parameters=case.dag.parameters,
        intermediates=[
            _operation_to_callable(
                transform.operation,
                backend=backend,
                broadcast_plan=broadcast_plan,
            )
            for transform in case.transforms
        ],
        return_name=case.dag.return_name,
    )


def _path_through_graph(
    implementation: Implementation,
    inputs: dict[str, object],
    output: str,
) -> tuple[BiltModelConstituent, ...]:
    ordered: list[BiltModelConstituent] = []
    visited: set[str] = set()
    visiting: list[str] = []

    def visit(name: str) -> None:
        if name in inputs or name in visited:
            return

        if name in visiting:
            cycle_start = visiting.index(name)
            cycle = (*visiting[cycle_start:], name)
            message = "Cyclic transform dependency: " + " -> ".join(repr(item) for item in cycle)
            raise BiltSystemEvaluationError(message)

        operation = implementation._components.get(name)  # noqa: SLF001

        if operation is None:
            message = f"No operation registered to produce output {name!r}"
            raise BiltSystemEvaluationError(message)

        visiting.append(name)

        for parameter in operation.__bilt_parameters__:
            visit(parameter)

        visiting.pop()
        visited.add(name)
        ordered.append(operation)

    visit(output)
    return tuple(ordered)


def _input_domain(
    value: object,
    registry: DomainRegistry,
) -> Domain:
    tensor = as_bilt_tensor(value)

    if isinstance(tensor, BiltTensor):
        tensor.observe(registry)

    return tensor.domain


def _operation_result_domain(
    operation: BiltModelConstituent,
    parameter_domains: tuple[Domain, ...],
) -> Domain:
    match operation:
        case BasicTransform():
            return _broadcast_domains(parameter_domains)

        case BasicEquation():
            message = "Equation evaluation is not yet implemented."
            raise NotImplementedError(message)

        case Interface():
            message = "Interface evaluation is not yet implemented."
            raise NotImplementedError(message)

        case _:
            message = f"Operation type {type(operation)} is not supported for domain analysis."
            raise NotImplementedError(message)


def _broadcast_domains(domains: tuple[Domain, ...]) -> Domain:
    dimensions = []

    for domain in domains:
        for dimension in domain.dims:
            if dimension not in dimensions:
                dimensions.append(dimension)

    return Domain(tuple(dimensions))


def _operation_to_callable(
    operation: BiltModelConstituent,
    *,
    backend: Backend,
    broadcast_plan: BroadcastPlan,
) -> BackendCallable:
    match operation:
        case BasicTransform():
            function_ast = typing.cast(
                "ast.FunctionDef",
                ast.parse(
                    textwrap.dedent(
                        inspect.getsource(
                            operation._function,  # noqa: SLF001
                        )
                    )
                ).body[0],
            )
            function_ast.decorator_list = [
                decorator
                for decorator in function_ast.decorator_list
                if not (isinstance(decorator, ast.Name) and decorator.id == "transform")
            ]

            function_ast = backend.apply_broadcast_plan(function_ast, broadcast_plan)
            return backend.transpile(function_ast)

        case BasicEquation():
            message = "Equation evaluation is not yet implemented."
            raise NotImplementedError(message)

        case Interface():
            message = "Interface evaluation is not yet implemented."
            raise NotImplementedError(message)

        case _:
            message = f"Operation type {type(operation)} is not supported for evaluation."
            raise NotImplementedError(message)

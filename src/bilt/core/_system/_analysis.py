__all__ = ["BiltSystemEvaluationError", "find_successful_case"]

import ast
import functools
import inspect
import textwrap
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


def validate_broadcasts(
    implementation: Implementation,
    inputs: dict[str, object],
    output: str,
    registry: DomainRegistry,
) -> Domain:
    """Validate named-domain broadcasting over the scalar transform DAG."""
    resolved: dict[str, Domain] = {}

    def resolve(name: str) -> Domain:
        existing = resolved.get(name)

        if existing is not None:
            return existing

        if name in inputs:
            value = as_bilt_tensor(inputs[name])

            if isinstance(value, BiltTensor):
                value.observe(registry)

            resolved[name] = value.domain
            return value.domain

        operation = implementation._components.get(name)  # noqa: SLF001

        if operation is None:
            message = f"No operation registered to produce output {name!r}"
            raise BiltSystemEvaluationError(message)

        parameter_domains = tuple(resolve(parameter) for parameter in operation.__bilt_parameters__)
        result = _broadcast_domains(parameter_domains)
        resolved[name] = result
        return result

    return resolve(output)


def _broadcast_domains(domains: tuple[Domain, ...]) -> Domain:
    dimensions = []

    for domain in domains:
        for dimension in domain.dims:
            if dimension not in dimensions:
                dimensions.append(dimension)

    return Domain(tuple(dimensions))


def find_successful_case(
    implementation: Implementation,
    inputs: dict[str, object],
    output: str,
    backend: Backend,
) -> BackendCase:
    """Find a successful case for the given inputs and output."""
    if output not in implementation._components:  # noqa: SLF001
        message = f"No operation registered to produce output {output!r}"
        raise BiltSystemEvaluationError(message)

    return BackendCase(
        name=output,
        parameters=tuple(inputs.keys()),
        intermediates=list(
            map(
                functools.partial(
                    _operation_to_callable,
                    backend=backend,
                ),
                (
                    implementation._components[x]  # noqa: SLF001
                    for x in _path_through_graph(
                        implementation,
                        inputs,
                        output,
                        backend,
                    )
                ),
            )
        ),
        return_name=output,
    )


def _path_through_graph(
    implementation: Implementation,
    inputs: dict[str, object],
    output: str,
    backend: Backend,
) -> list[str]:
    ordered_callables: list[str] = []
    operation = implementation._components[output]  # noqa: SLF001

    for parameter in operation.__bilt_parameters__:
        if parameter not in inputs:
            ordered_callables.extend(
                _path_through_graph(
                    implementation,
                    inputs,
                    parameter,
                    backend,
                )
            )

    ordered_callables.append(output)

    return list(dict.fromkeys(ordered_callables))


def _operation_to_callable(
    operation: BiltModelConstituent,
    *,
    backend: Backend,
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
            return backend.transpile(function_ast)

        case BasicEquation():
            msg = "Equation evaluation is not yet implemented."
            raise NotImplementedError(msg)

        case Interface():
            msg = "Interface evaluation is not yet implemented."
            raise NotImplementedError(msg)

        case _:
            msg = f"Operation type {type(operation)} is not supported for evaluation."
            raise NotImplementedError(msg)

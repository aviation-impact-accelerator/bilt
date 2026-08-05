__all__ = [
    "BroadcastValidationError",
    "ValidatedAnalysisPlan",
    "ValidatedPlanStep",
    "validate_broadcasts",
]

import dataclasses
import types
import typing

from bilt.core._equation import BasicEquation
from bilt.core._error import BiltCoreError
from bilt.core._interface import Interface
from bilt.core._transform import BasicTransform
from bilt.symbolic._domains import Domain

if typing.TYPE_CHECKING:
    from bilt.core._structural_analysis._plan import (
        AnalysisPlan,
        PlanStep,
    )


class BroadcastValidationError(BiltCoreError):
    """Raised when domains cannot be propagated through a plan."""


@typing.runtime_checkable
class _DomainValue(typing.Protocol):
    """A value carrying a named domain."""

    @property
    def domain(self) -> Domain: ...


@dataclasses.dataclass(frozen=True, slots=True)
class ValidatedPlanStep:
    """One plan step annotated with its input and output domains."""

    step: PlanStep
    parameters: tuple[str, ...]
    parameter_domains: tuple[Domain, ...]
    result_domain: Domain


@dataclasses.dataclass(frozen=True, slots=True)
class ValidatedAnalysisPlan:
    """An analysis plan with named-domain broadcasts resolved."""

    plan: AnalysisPlan
    steps: tuple[ValidatedPlanStep, ...]
    value_domains: types.MappingProxyType[str, Domain]

    @property
    def result_domain(self) -> Domain:
        return self.value_domains[self.plan.output]


def validate_broadcasts(
    plan: AnalysisPlan,
    inputs: dict[str, object],
) -> ValidatedAnalysisPlan:
    """Propagate named domains through an analysis plan."""
    resolved = {name: _input_domain(value) for name, value in inputs.items()}
    validated_steps: list[ValidatedPlanStep] = []

    for step in plan.steps:
        parameters = _step_parameters(step)

        try:
            parameter_domains = tuple(resolved[parameter] for parameter in parameters)
        except KeyError as error:
            parameter = typing.cast("str", error.args[0])
            message = f"Plan step for output {step.output!r} depends on unresolved value {parameter!r}."
            raise BroadcastValidationError(message) from None

        result_domain = _operation_result_domain(
            step,
            parameter_domains,
        )
        resolved[step.output] = result_domain

        validated_steps.append(
            ValidatedPlanStep(
                step=step,
                parameters=parameters,
                parameter_domains=parameter_domains,
                result_domain=result_domain,
            )
        )

    if plan.output not in resolved:
        message = f"Analysis plan did not resolve output {plan.output!r}."
        raise BroadcastValidationError(message)

    return ValidatedAnalysisPlan(
        plan=plan,
        steps=tuple(validated_steps),
        value_domains=types.MappingProxyType(dict(resolved)),
    )


def _input_domain(value: object) -> Domain:
    if isinstance(value, _DomainValue):
        return value.domain

    return Domain.scalar()


def _step_parameters(step: PlanStep) -> tuple[str, ...]:
    return tuple(parameter for parameter in step.operation.__bilt_parameters__ if parameter != step.output)


def _operation_result_domain(
    step: PlanStep,
    parameter_domains: tuple[Domain, ...],
) -> Domain:
    match step.operation:
        case BasicTransform():
            return _broadcast_domains(parameter_domains)

        case BasicEquation():
            message = "Equation domain analysis is not yet implemented."
            raise NotImplementedError(message)

        case Interface():
            message = "Interface domain analysis is not yet implemented."
            raise NotImplementedError(message)

        case _:
            message = f"Operation type {type(step.operation)} is not supported for domain analysis."
            raise NotImplementedError(message)


def _broadcast_domains(
    domains: tuple[Domain, ...],
) -> Domain:
    dimensions = []

    for domain in domains:
        for dimension in domain.dims:
            if dimension not in dimensions:
                dimensions.append(dimension)

    return Domain(tuple(dimensions))

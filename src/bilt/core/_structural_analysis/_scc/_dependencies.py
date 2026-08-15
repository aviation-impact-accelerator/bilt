__all__ = [
    "required_inputs_for_plan",
    "required_inputs_for_step",
    "sub_system_plan_for_step",
]

from bilt.core._equation import BasicEquation
from bilt.core._structural_analysis._errors import BiltSystemAnalysisError
from bilt.core._structural_analysis._plan import (
    AnalysisPlan,
    CoupledBlock,
    PlanStep,
    SubSystemPlan,
)
from bilt.core._sub_system import SubSystemConstituent
from bilt.core._transform import BasicTransform


def sub_system_plan_for_step(
    plan_step: PlanStep,
    *,
    available_child_ports: set[str] | None = None,
) -> SubSystemPlan | None:
    """Return the child plan selected for a sub-system step."""
    if not isinstance(plan_step.operation, SubSystemConstituent):
        msg = f"{plan_step.operation!r} is not a sub-system constituent."
        raise BiltSystemAnalysisError(msg)

    if plan_step.sub_system_plan is not None:
        return plan_step.sub_system_plan

    sub_system = plan_step.operation
    child_output = sub_system.unqualify(plan_step.output)

    if child_output is None:
        msg = f"{sub_system!r} does not have port {plan_step.output!r}."
        raise BiltSystemAnalysisError(msg)

    if available_child_ports is None:
        available_child_ports = set(sub_system.interface.__bilt_parameters__) - {child_output}
    else:
        available_child_ports = set(available_child_ports) - {child_output}

    child_plan = sub_system.implementation.analyser.plan_for_target(
        available_child_ports,
        child_output,
    )

    if child_plan is None:
        return None

    child_inputs = required_inputs_for_plan(child_plan)

    return SubSystemPlan(
        plan=child_plan,
        inputs=frozenset(sub_system.qualify(port) for port in child_inputs),
    )


def required_inputs_for_step(
    plan_step: PlanStep,
) -> frozenset[str] | None:
    """Return the inputs required to structurally realise a plan step."""
    match plan_step.operation:
        case BasicTransform() as transform:
            if plan_step.output != transform.__bilt_name__:
                msg = f"{transform!r} does not produce {plan_step.output!r}."
                raise BiltSystemAnalysisError(msg)

            return frozenset(transform.__bilt_parameters__)

        case BasicEquation() as equation:
            if plan_step.output not in equation.__bilt_parameters__:
                msg = f"{equation!r} does not have parameter {plan_step.output!r}."
                raise BiltSystemAnalysisError(msg)

            return frozenset(equation.__bilt_parameters__) - {plan_step.output}

        case SubSystemConstituent():
            sub_system_plan = sub_system_plan_for_step(plan_step)

            if sub_system_plan is None:
                return None

            return sub_system_plan.inputs

        case _:
            msg = f"Constituent type {type(plan_step.operation)} is not supported."
            raise NotImplementedError(msg)


def required_inputs_for_plan(
    plan: AnalysisPlan,
) -> frozenset[str]:
    """Return the external inputs consumed by an analysis plan."""
    produced: set[str] = set()
    required: set[str] = set()

    for item in plan.steps:
        match item:
            case PlanStep() as step:
                produced.add(step.output)

            case CoupledBlock() as block:
                produced.update(block.outputs)

    for item in plan.steps:
        match item:
            case PlanStep() as step:
                inputs = required_inputs_for_step(step)

                if inputs is not None:
                    required.update(inputs)

            case CoupledBlock() as block:
                for step in block.steps:
                    inputs = required_inputs_for_step(step)

                    if inputs is not None:
                        required.update(inputs)

    return frozenset(required - produced)

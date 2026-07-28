from __future__ import annotations

import typing

from bilt.core._equation import BasicEquation
from bilt.core._structural_analysis._errors import BiltSystemEvaluationError
from bilt.core._structural_analysis._plan import AnalysisPlan, PlanStep
from bilt.core._transform import BasicTransform

if typing.TYPE_CHECKING:
    from bilt.core._model_constituent import BiltModelConstituent


def exhaustive_structural_backend(
    constituents: tuple[BiltModelConstituent, ...],
    inputs: frozenset[str],
    output: str,
) -> AnalysisPlan:
    """Brute-force DFS backend that finds a valid topological execution order.

    We arrange all model constituents into a mapping of possible ouputs to the constituents that can produce them.
    We then look for the target output and backtrack to find a path which ends with valid inputs.
    If a valid path is found, we return the ordered steps as an AnalysisPlan.
    """
    output_to_constituent = output_to_constituent_map(constituents)

    if output not in output_to_constituent:
        message = f"No operation registered to produce output {output!r}"
        raise BiltSystemEvaluationError(message)

    available: set[str] = set(inputs)
    steps: list[PlanStep] = []

    if not _dfs(output_to_constituent, available, output, steps):
        message = f"No valid execution path found for output {output!r} given inputs {inputs!r}"
        raise BiltSystemEvaluationError(message)

    return AnalysisPlan(inputs=frozenset(collect_used_inputs(steps)), steps=tuple(steps), output=output)


def output_to_constituent_map(constituents: tuple[BiltModelConstituent, ...]) -> dict[str, list[BiltModelConstituent]]:
    """Create a mapping of possible outputs to the constituents that can produce them."""
    candidate_map: dict[str, list[BiltModelConstituent]] = {}
    for component in constituents:
        match component:
            case BasicEquation():
                for param in component.__bilt_parameters__:
                    candidate_map.setdefault(param, []).append(component)
            case _:
                candidate_map.setdefault(component.__bilt_name__, []).append(component)
    return candidate_map


def _dfs(
    output_to_constituent_map: dict[str, list[BiltModelConstituent]],
    available: set[str],
    output: str,
    steps: list[PlanStep],
    visiting: frozenset[str] = frozenset(),
) -> bool:
    if output in visiting:
        return False

    visiting = visiting | {output}

    for component in output_to_constituent_map.get(output, []):
        required = extract_required_parameters(component, output)

        checkpoint_len = len(steps)
        checkpoint_available = frozenset(available)

        if all(
            param in available or _dfs(output_to_constituent_map, available, param, steps, visiting)
            for param in required
        ):
            steps.append(PlanStep(operation=component, output=output))
            available.add(output)
            return True

        del steps[checkpoint_len:]
        available.clear()
        available.update(checkpoint_available)

    return False


def extract_required_parameters(component: BiltModelConstituent, output: str) -> list[str]:
    """Extract the required parameters for a given component.

    Equation requires all parameters except the output, transform requires all parameters.
    """
    match component:
        case BasicEquation():
            return [p for p in component.__bilt_parameters__ if p != output]
        case BasicTransform():
            return list(component.__bilt_parameters__)
        case _:
            msg = f"Unsupported component type: {type(component)} for structural analysis."
            raise NotImplementedError(msg)


def collect_used_inputs(steps: list[PlanStep]) -> set[str]:
    """Collect all used inputs from the steps."""
    used_inputs: set[str] = set()
    for step in steps:
        used_inputs.update(extract_required_parameters(step.operation, step.output))
    for step in steps:
        used_inputs.discard(step.output)

    return used_inputs

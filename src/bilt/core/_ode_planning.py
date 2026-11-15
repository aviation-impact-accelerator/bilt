"""Structural multi-output ODE planning against Bilt's existing SCC machinery.

The runtime sees only two already-compiled *pure instantaneous* operations:
one for all initial conditions and one for all requested state rates.
"""

import copy
import typing

from bilt._backend._python import PYTHON_BACKEND
from bilt.core._equation import BasicEquation
from bilt.core._evaluation_request import EvaluationRequest
from bilt.core._port import mangle_name
from bilt.core._structural_analysis._causalization_block_solver._backend import CausalizationStructuralAnalyzer
from bilt.core._structural_analysis._causalization_block_solver._full_incidence_graph import FullIncidenceGraph
from bilt.core._structural_analysis._causalization_block_solver._structural_scope import StructuralAnalysis
from bilt.core._structural_analysis._causalization_block_solver._structural_validation import validate_structure
from bilt.core._structural_analysis._realize import BlockRealizationFailure, RealizationFailure, realize_plan
from bilt.symbolic._domains import Domain
from bilt.symbolic._expressions import symbol_expr
from bilt.symbolic._tree import equality_tree, symbol_tree
from bilt.symbolic._walk import subtrees
from bilt.symbolic._atoms import SymbolType

if typing.TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping

    from bilt._backend._callable import BackendCallable
    from bilt.core._model_constituent import BiltModelConstituent
    from bilt.core._ode_prototype import _State
    from bilt.symbolic._tree import Tree


def rate_name(name: str) -> str:
    return f"__bilt_ode_rate_{name}"


def initial_name(name: str) -> str:
    return f"__bilt_ode_initial_{name}"


def location_name(name: str) -> str:
    return f"__bilt_ode_location_{name}"


def compile_ode_plans(
    states: tuple[_State, ...],
    components: tuple[BiltModelConstituent, ...],
    *,
    input_names: set[str],
    coordinate: str,
) -> tuple[Callable[[Mapping[str, float]], tuple[object, ...]], Callable[[Mapping[str, float]], tuple[object, ...]]]:
    """Build coherent instantaneous RHS and initial-value plans, independently."""
    if any(name.startswith("__bilt_ode_") for name in input_names | {state.name for state in states}):
        msg = "The __bilt_ode_ prefix is reserved for internal ODE symbols."
        raise ValueError(msg)

    ordinary = tuple(
        item
        for item in components
        if not isinstance(item, BasicEquation)
        or not any(
            node.head.value in ("deferred_grad", "at")
            for node in subtrees(item.__bilt_equation_tree__)
            if not node.is_leaf
        )
    )
    rates = tuple(
        _rewrite(
            state.source,
            equality_tree(symbol_tree(rate_name(state.name)), state.rhs),
        )
        for state in states
    )
    initials = tuple(
        _rewrite(
            state.condition,
            equality_tree(symbol_tree(initial_name(state.name)), state.initial),
        )
        for state in states
    )
    locations = tuple(
        _rewrite(
            state.condition,
            equality_tree(symbol_tree(location_name(state.name)), state.initial_coordinate),
        )
        for state in states
    )

    rhs_outputs = tuple(rate_name(state.name) for state in states)
    initial_outputs = tuple(initial_name(state.name) for state in states) + tuple(
        location_name(state.name) for state in states
    )
    rhs_inputs = input_names | {state.name for state in states} | {coordinate}

    # Traverse the ORIGINAL differential equations with states still unknown.
    # This retains every connected algebraic row, including constraints that
    # would disappear if current state values were treated as inputs first.
    original_scope = StructuralAnalysis(components).inspect(
        input_names | {coordinate}, {state.name for state in states}
    )
    if original_scope.boundaries:
        msg = "Cross-subsystem differential constraints require a qualified ODE planner."
        raise ValueError(msg)
    rate_rewrites = {id(state.source): rate for state, rate in zip(states, rates, strict=True)}
    conditions = {id(state.condition) for state in states}
    full_rhs = tuple(
        rate_rewrites.get(id(component), component)
        for component in original_scope.constraints
        if id(component) not in conditions
    )
    rhs = _compile_many(full_rhs, rhs_outputs, rhs_inputs, validate_full_scope=True)
    init = _compile_many((*ordinary, *initials, *locations), initial_outputs, input_names)
    return rhs, init


def _rewrite(source: BasicEquation, tree: Tree) -> BasicEquation:
    """Preserve annotations and source identity when replacing a formal node."""
    equation = copy.copy(source)
    names = tuple(
        sorted(
            {
                typing.cast("str", node.head.value)
                for node in subtrees(tree)
                if node.is_leaf and node.head.atom_type is SymbolType
            }
        )
    )
    equation.__bilt_equation_tree__ = tree
    equation.__bilt_parameters__ = names
    equation.__bilt_equation_symbols__ = {name: symbol_expr(name) for name in names}
    equation.__bilt_port_declarations__ = {
        name: declaration for name, declaration in source.__bilt_port_declarations__.items() if name in names
    }
    return equation


def _compile_many(
    components: tuple[BiltModelConstituent, ...],
    outputs: tuple[str, ...],
    available: set[str],
    *,
    validate_full_scope: bool = False,
) -> Callable[[Mapping[str, float]], tuple[object, ...]]:
    """Select a single matching and compile its shared SCC dependency closure."""
    scope = StructuralAnalysis(components).inspect(available, set(outputs))
    if scope.boundaries:
        msg = "Differential subsystem coupling requires the full ODE boundary planner."
        raise ValueError(msg)
    # A differential scope must validate every connected equation, not only
    # the particular route selected to compute the requested rate outputs.
    incidence = FullIncidenceGraph(components if validate_full_scope else scope.constraints)
    known = frozenset(
        incidence.attribute_nodes[name] for name in available if name in incidence.attribute_nodes
    )
    validate_structure(incidence.graph, frozenset(incidence.attribute_nodes.values()), known)
    request = EvaluationRequest(outputs[0], {name: Domain.scalar() for name in available})
    analyser = CausalizationStructuralAnalyzer(scope.constraints, PYTHON_BACKEND)
    plan = analyser.plan_many(request, outputs)
    case = realize_plan(plan, PYTHON_BACKEND)
    if isinstance(case, (RealizationFailure, BlockRealizationFailure)):
        msg = "ODE plan could not be lowered to a supported instantaneous operation."
        raise ValueError(msg) from case.cause

    def evaluate(values: Mapping[str, float]) -> tuple[object, ...]:
        current: dict[str, object] = {mangle_name(name): value for name, value in values.items()}
        for intermediate in case.intermediates:
            current[intermediate.name] = intermediate.evaluate(current)
        return tuple(current[mangle_name(name)] for name in outputs)

    return evaluate

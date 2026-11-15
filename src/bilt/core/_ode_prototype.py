"""Experimental explicit ODE evaluation from deferred symbolic grad nodes.

This deliberately narrow integration spike supports local first-order IVPs
whose RHSs are explicit expressions of scalar states and known parameters.
It does not replace the archived ODE prototype's complete SCC planner.
"""

import dataclasses
import math
import typing

import numpy as np

from bilt._backend._python import PYTHON_BACKEND
from bilt.core._equation import BasicEquation
from bilt.core._evaluation_context import EvaluationContext
from bilt.core._ode_planning import compile_ode_plans
from bilt.core._tensors._tensors import Tensor
from bilt.symbolic._atoms import SymbolType
from bilt.symbolic._differential import AT, DEFERRED_GRAD
from bilt.symbolic._walk import subtrees

if typing.TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from bilt._backend import Backend
    from bilt.core._model_constituent import BiltModelConstituent
    from bilt.symbolic._tree import Tree


@dataclasses.dataclass(frozen=True)
class _State:
    name: str
    coordinate: str
    rhs: Tree
    initial: Tree
    initial_coordinate: Tree
    source: BasicEquation
    condition: BasicEquation


class ExplicitODEPrototype:
    """Classify and execute only unambiguous scalar instantaneous ODE systems."""

    def __init__(self, states: tuple[_State, ...], components: tuple[BiltModelConstituent, ...]) -> None:
        self.states = states
        self.components = components
        self.coordinate = states[0].coordinate
        self.state_names = frozenset(state.name for state in states)

    @classmethod
    def extract(  # noqa: C901, PLR0912
        cls, components: Iterable[BiltModelConstituent]
    ) -> ExplicitODEPrototype | None:
        components = tuple(components)
        rates: dict[str, tuple[str, Tree, BasicEquation]] = {}
        initial: dict[str, tuple[str, Tree, Tree, BasicEquation]] = {}
        for component in components:
            if not isinstance(component, BasicEquation):
                continue
            tree = component.__bilt_equation_tree__
            operations = [node for node in subtrees(tree) if node.head in (DEFERRED_GRAD, AT)]
            if not operations:
                continue
            if not tree.is_equality or len(operations) != 1:
                msg = "ODE prototype requires one formal derivative or point condition per equality."
                raise ValueError(msg)
            node = operations[0]
            if tree.left is node:
                expression = tree.right
            elif tree.right is node:
                expression = tree.left
            else:
                msg = "A formal derivative or point condition must occupy one whole side of its equality."
                raise ValueError(msg)
            if any(child.head in (DEFERRED_GRAD, AT) for child in subtrees(expression)):
                msg = "Nested formal differential operators are unsupported."
                raise ValueError(msg)
            state = _symbol(node.args[0])
            coordinate = _symbol(node.args[1])
            if state == coordinate:
                msg = "An ODE state and its independent coordinate must be different ports."
                raise ValueError(msg)
            if node.head is DEFERRED_GRAD:
                if state in rates:
                    msg = f"Duplicate rate equations for {state!r}."
                    raise ValueError(msg)
                rates[state] = (coordinate, expression, component)
            else:
                if state in initial:
                    msg = f"Duplicate initial conditions for {state!r}."
                    raise ValueError(msg)
                initial[state] = (coordinate, node.args[2], expression, component)
        if not rates and not initial:
            return None
        if set(rates) != set(initial):
            msg = f"Each evolving state needs exactly one rate and initial condition: rates={sorted(rates)}, initial={sorted(initial)}."
            raise ValueError(msg)
        coordinates = {value[0] for value in rates.values()} | {value[0] for value in initial.values()}
        if len(coordinates) != 1:
            msg = "All ODE states and their conditions must use one common coordinate."
            raise ValueError(msg)
        states = tuple(
            _State(
                name,
                rates[name][0],
                rates[name][1],
                initial[name][2],
                initial[name][1],
                rates[name][2],
                initial[name][3],
            )
            for name in sorted(rates)
        )
        return cls(states, components)

    def solve(  # noqa: C901, PLR0912, PLR0915
        self, inputs: Mapping[str, object], backend: Backend
    ) -> dict[str, Tensor]:
        """Run one shared SciPy RK45 solve for all requested state trajectories."""
        if backend is not PYTHON_BACKEND:
            msg = "The experimental ODE runtime currently supports only the Python backend."
            raise NotImplementedError(msg)

        coordinate = inputs.get(self.coordinate)
        if not isinstance(coordinate, Tensor) or len(coordinate.domain.dims) != 1:
            msg = f"ODE independent coordinate {self.coordinate!r} requires a one-dimensional Tensor."
            raise ValueError(msg)
        if self.state_names & inputs.keys():
            msg = "ODE state trajectories must not be supplied as boundary inputs."
            raise ValueError(msg)
        if any(isinstance(value, Tensor) for name, value in inputs.items() if name != self.coordinate):
            msg = "The explicit ODE prototype supports scalar boundary parameters only."
            raise ValueError(msg)
        times = np.asarray(coordinate.values, dtype=float)
        if not times.size or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
            msg = "ODE output coordinates must be finite and strictly increasing."
            raise ValueError(msg)

        parameters: dict[str, float] = {}
        for name, value in inputs.items():
            if name == self.coordinate:
                continue
            if not isinstance(value, (int, float, np.number)):
                msg = "The experimental ODE prototype requires plain numeric scalar parameters."
                raise TypeError(msg)
            parameters[name] = float(value)

        # The archived solver's full-scope validation and shared matching now
        # determine the RHS and initialization, including algebraic intermediates.
        with EvaluationContext():
            rhs_plan, initial_plan = compile_ode_plans(
                self.states,
                self.components,
                input_names=set(parameters),
                coordinate=self.coordinate,
            )
        initial_results = initial_plan(parameters)
        count = len(self.states)
        initial = np.asarray([_scalar(value) for value in initial_results[:count]], dtype=float)
        locations = [_scalar(value) for value in initial_results[count:]]
        if len(set(locations)) != 1:
            msg = "ODE states must share a common initial coordinate."
            raise ValueError(msg)
        t0 = locations[0]
        if times[0] < t0:
            msg = "Backward ODE integration is unsupported in this prototype."
            raise ValueError(msg)

        def rhs(time: float, y: np.ndarray) -> np.ndarray:
            values = dict(parameters)
            values[self.coordinate] = time
            values.update(zip((state.name for state in self.states), y, strict=True))
            return np.asarray([_scalar(value) for value in rhs_plan(values)], dtype=float)

        if times[-1] == t0:
            result = initial[:, None]
        else:
            try:
                from scipy.integrate import solve_ivp  # noqa: PLC0415
            except ImportError as error:
                msg = "Experimental ODE integration requires SciPy."
                raise ImportError(msg) from error
            solution = solve_ivp(
                rhs,
                (t0, float(times[-1])),
                initial,
                t_eval=times,
                rtol=1e-9,
                atol=1e-11,
                max_step=0.05,
            )
            if not solution.success:
                msg = f"ODE integration failed: {solution.message}"
                raise RuntimeError(msg)
            result = solution.y
        if result.shape != (len(self.states), len(times)) or not np.isfinite(result).all():
            msg = "ODE returned an invalid or nonfinite trajectory."
            raise RuntimeError(msg)
        return {
            state.name: Tensor(values=result[index], domain=coordinate.domain)
            for index, state in enumerate(self.states)
        }


def _symbol(tree: Tree) -> str:
    if not tree.is_leaf or tree.head.atom_type is not SymbolType:
        msg = "Formal derivatives and point conditions require named state and coordinate ports."
        raise ValueError(msg)
    return typing.cast("str", tree.head.value)


def _symbols(tree: Tree) -> set[str]:
    return {
        typing.cast("str", node.head.value)
        for node in subtrees(tree)
        if node.is_leaf and node.head.atom_type is SymbolType
    }



def _scalar(value: object) -> float:
    array = np.asarray(value)
    if array.size != 1:
        msg = "An ODE instantaneous expression must evaluate to a scalar."
        raise ValueError(msg)
    number = float(array.item())
    if not math.isfinite(number):
        msg = "An ODE instantaneous expression returned a nonfinite value."
        raise ValueError(msg)
    return number

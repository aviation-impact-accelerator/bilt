"""Prototype: optimise a lowered, named-domain Bilt calculation with asdex and IPOPT.

Bilt performs its structural analysis, unit erasure, domain alignment and JAX
lowering once. Sparse differentiation and optimisation then see only compiled
JAX functions accepting numerical arrays, with no Bilt system evaluation in
the differentiated function.
"""

import importlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, cast

import jax
import jax.numpy as jnp
import numpy as np

from bilt import JAX_BACKEND, Implementation, Interface, Tensor, dim, equation, field, system, tensor
from bilt.core._evaluation_context import EvaluationContext
from bilt.core._evaluation_request import EvaluationRequest
from bilt.core._port import mangle_inputs
from bilt.core._system._lowering_and_lifting import lower_inputs
from bilt.core._tensors._tensor_alignment import generate_alignment_plan
from bilt.symbolic._domains import Domain

if TYPE_CHECKING:
    from bilt._backend._callable import BackendCallable

# Dimensionless toy example: each leg's unconstrained optimum is speed = 1.
DISTANCES = (1.0, 1.2, 0.8, 1.6, 1.1, 1.4, 0.9, 1.3)
REQUIRED_SPEEDS = (0.8, 1.15, 0.95, 1.3, 1.1, 0.9, 1.45, 1.0)


@system
class CruiseLeg(metaclass=Interface):
    """Nondimensional aerodynamic mission leg, evaluated over a named domain."""

    speed: float = field()
    distance: float = field()
    parasite_coefficient: float = field()
    induced_coefficient: float = field()
    drag: float = field()
    energy: float = field()
    time: float = field()


@equation
def drag_balance(
    drag: float,
    speed: float,
    parasite_coefficient: float,
    induced_coefficient: float,
) -> bool:
    """Relate drag and speed through the parasite and induced terms."""
    return drag == parasite_coefficient * speed**2 + induced_coefficient / speed**2


@equation
def energy_balance(energy: float, drag: float, distance: float) -> bool:
    """Relate leg energy to aerodynamic drag and distance."""
    return energy == drag * distance


@equation
def travel_time_balance(time: float, distance: float, speed: float) -> bool:
    """Relate journey time to distance and cruise speed."""
    return distance == speed * time


cruise_model = Implementation(CruiseLeg, [drag_balance, energy_balance, travel_time_balance])


@dataclass(frozen=True)
class CompiledCruise:
    """Only backend numerical callables and already lowered fixed inputs."""

    energy: BackendCallable
    time: BackendCallable
    fixed_inputs: dict[str, jax.Array]

    def __call__(self, speeds: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Run the JAX numerical graph; no Bilt structural analysis occurs here."""
        inputs = {**self.fixed_inputs, "speed": speeds}
        return cast("jax.Array", self.energy.evaluate(inputs)), cast("jax.Array", self.time.evaluate(inputs))


def compile_cruise_legs(distances: np.ndarray) -> CompiledCruise:
    """Perform Bilt analysis, tensor alignment and lowering once, outside JAX AD.

    The Bilt input tensors are concrete and carry the named 'leg' domain.
    Only their lowered numerical arrays are passed to compiled callables.
    This private API is used only to explore a future public compile interface.
    """
    distances = np.asarray(distances, dtype=float)
    leg = dim("leg")
    model_inputs: dict[str, object] = {
        "speed": tensor(np.ones_like(distances), over=leg),
        "distance": tensor(distances, over=leg),
        "parasite_coefficient": 1.0,
        "induced_coefficient": 1.0,
    }

    with EvaluationContext() as evaluation:
        evaluation.domains.bind_all(model_inputs.values())
        input_domains = {
            name: value.domain if isinstance(value, Tensor) else Domain.scalar() for name, value in model_inputs.items()
        }
        energy = cruise_model.get_evaluation(
            EvaluationRequest(output="energy", input_domains=input_domains), JAX_BACKEND
        )
        time = cruise_model.get_evaluation(EvaluationRequest(output="time", input_domains=input_domains), JAX_BACKEND)

        aligned = JAX_BACKEND.apply_alignment(
            lower_inputs(model_inputs),
            generate_alignment_plan(model_inputs),
        )

    lowered_inputs = mangle_inputs(aligned)
    fixed_inputs = {name: jnp.asarray(value) for name, value in lowered_inputs.items() if name != "speed"}
    return CompiledCruise(energy=energy, time=time, fixed_inputs=fixed_inputs)


@dataclass(frozen=True)
class OptimisationResult:
    """Solution and derivative-sparsity diagnostics."""

    speeds: np.ndarray
    total_energy: float
    jacobian_nonzeros: int | None
    hessian_lower_nonzeros: int | None
    max_time_violation: float


def optimise_cruise_speeds(
    *, solver: Literal["ipopt", "pounce-asdex", "pounce-jax"] = "ipopt"
) -> OptimisationResult:  # noqa: C901, PLR0912
    """Minimise Bilt cruise energy with one of three numerical solver paths.

    All paths share Bilt's compiled, named-domain JAX objective and constraints.
    The first two use asdex sparse derivatives through a cyipopt-style API;
    the third asks POUNCE to construct its own coloured sparse derivatives.
    """
    # IPOPT-style interior-point methods use double precision.
    jax.config.update("jax_enable_x64", True)  # noqa: FBT003

    distances = jnp.asarray(DISTANCES)
    required = jnp.asarray(REQUIRED_SPEEDS)
    time_limits = distances / required
    n = len(DISTANCES)

    # Bilt performs structural analysis and JAX lowering exactly once, here.
    compiled = compile_cruise_legs(np.asarray(DISTANCES))

    def objective(speeds: jax.Array) -> jax.Array:
        energies, _ = compiled(speeds)
        return jnp.sum(energies)

    def constraints(speeds: jax.Array) -> jax.Array:
        _, times = compiled(speeds)
        return times - time_limits

    def lagrangian(speeds: jax.Array, multipliers: jax.Array, factor: jax.Array) -> jax.Array:
        return factor * objective(speeds) + jnp.dot(multipliers, constraints(speeds))

    x0 = jnp.maximum(required, 1.2)
    objective_fn = jax.jit(objective)
    bounds = {
        "n": n,
        "m": n,
        "lb": np.full(n, 0.6),
        "ub": np.full(n, 2.0),
        "cl": np.full(n, -1e20),
        "cu": np.zeros(n),
    }

    if solver == "pounce-jax":
        # POUNCE's own coloured JAX derivative implementation, without asdex.
        pounce_jax = importlib.import_module("pounce.jax")
        problem = pounce_jax.from_jax(
            objective,
            constraints,
            **bounds,
            sparse=True,
            x0=np.asarray(x0),
        )
        jacobian_nonzeros = None
        hessian_lower_nonzeros = None
    elif solver in ("ipopt", "pounce-asdex"):
        asdex = importlib.import_module("asdex")
        gradient_fn = jax.jit(jax.grad(objective))
        constraints_fn = jax.jit(constraints)

        # Both solvers receive identical asdex sparsity and derivative values.
        jacobian_fn = jax.jit(asdex.jacobian(constraints, x0))
        hessian_fn = jax.jit(asdex.hessian(lagrangian, x0, jnp.zeros(n), jnp.asarray(1.0)))

        jacobian_example = jacobian_fn(x0)
        jac_rows, jac_cols = np.asarray(jacobian_example.indices).T
        hessian_example = hessian_fn(x0, jnp.zeros(n), jnp.asarray(1.0))
        hess_rows_full, hess_cols_full = np.asarray(hessian_example.indices).T
        lower = hess_rows_full >= hess_cols_full
        hess_rows = hess_rows_full[lower]
        hess_cols = hess_cols_full[lower]

        class Callbacks:
            """Standard cyipopt-compatible callbacks for either Rust or IPOPT."""

            def objective(self, x: np.ndarray) -> float:
                return float(objective_fn(jnp.asarray(x)))

            def gradient(self, x: np.ndarray) -> np.ndarray:
                return np.asarray(gradient_fn(jnp.asarray(x)))

            def constraints(self, x: np.ndarray) -> np.ndarray:
                return np.asarray(constraints_fn(jnp.asarray(x)))

            def jacobianstructure(self) -> tuple[np.ndarray, np.ndarray]:
                return jac_rows, jac_cols

            def jacobian(self, x: np.ndarray) -> np.ndarray:
                return np.asarray(jacobian_fn(jnp.asarray(x)).data)

            def hessianstructure(self) -> tuple[np.ndarray, np.ndarray]:
                return hess_rows, hess_cols

            def hessian(self, x: np.ndarray, lagrange: np.ndarray, obj_factor: float) -> np.ndarray:
                return np.asarray(hessian_fn(jnp.asarray(x), jnp.asarray(lagrange), jnp.asarray(obj_factor)).data)[lower]

        library = importlib.import_module("cyipopt" if solver == "ipopt" else "pounce")
        problem = library.Problem(problem_obj=Callbacks(), **bounds)
        jacobian_nonzeros = len(jac_rows)
        hessian_lower_nonzeros = len(hess_rows)
    else:
        msg = f"Unknown optimisation solver: {solver!r}"
        raise ValueError(msg)

    problem.add_option("tol", 1e-9)
    problem.add_option("print_level", 0)
    problem.add_option("sb", "yes")

    speeds, info = problem.solve(np.asarray(x0))
    if info["status"] not in (0, 1):
        msg = f"{solver} did not converge: {info['status_msg']}"
        raise RuntimeError(msg)

    _, solved_times = compiled(jnp.asarray(speeds))
    return OptimisationResult(
        speeds=np.asarray(speeds),
        total_energy=float(objective_fn(jnp.asarray(speeds))),
        jacobian_nonzeros=jacobian_nonzeros,
        hessian_lower_nonzeros=hessian_lower_nonzeros,
        max_time_violation=float(np.maximum(np.asarray(solved_times - time_limits), 0).max()),
    )

if __name__ == "__main__":
    result = optimise_cruise_speeds()
    expected = np.maximum(1.0, np.asarray(REQUIRED_SPEEDS))
    np.testing.assert_allclose(result.speeds, expected, atol=1e-5, rtol=0)

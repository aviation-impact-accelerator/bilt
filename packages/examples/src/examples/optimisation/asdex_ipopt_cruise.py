"""Prototype: optimise a lowered, named-domain Bilt calculation with asdex and IPOPT.

Bilt performs its structural analysis, unit erasure, domain alignment and JAX
lowering once. Sparse differentiation and optimisation then see only compiled
JAX functions accepting numerical arrays, with no Bilt system evaluation in
the differentiated function.
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from bilt import Implementation, Interface, JAX_BACKEND, Tensor, dim, equation, field, system, tensor
from bilt._backend._callable import BackendCallable
from bilt.core._evaluation_context import EvaluationContext
from bilt.core._evaluation_request import EvaluationRequest
from bilt.core._port import mangle_inputs
from bilt.core._system._lowering_and_lifting import lower_inputs
from bilt.core._tensors._tensor_alignment import generate_alignment_plan
from bilt.symbolic._domains import Domain

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
    return drag == parasite_coefficient * speed**2 + induced_coefficient / speed**2


@equation
def energy_balance(energy: float, drag: float, distance: float) -> bool:
    return energy == drag * distance


@equation
def travel_time_balance(time: float, distance: float, speed: float) -> bool:
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
        return self.energy.evaluate(inputs), self.time.evaluate(inputs)


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
            name: value.domain if isinstance(value, Tensor) else Domain.scalar()
            for name, value in model_inputs.items()
        }
        energy = cruise_model.get_evaluation(EvaluationRequest(output="energy", input_domains=input_domains), JAX_BACKEND)
        time = cruise_model.get_evaluation(EvaluationRequest(output="time", input_domains=input_domains), JAX_BACKEND)

        aligned = JAX_BACKEND.apply_alignment(
            lower_inputs(model_inputs),
            generate_alignment_plan(model_inputs),
        )

    lowered_inputs = mangle_inputs(aligned)
    fixed_inputs = {
        name: jnp.asarray(value)
        for name, value in lowered_inputs.items()
        if name != "speed"
    }
    return CompiledCruise(energy=energy, time=time, fixed_inputs=fixed_inputs)


@dataclass(frozen=True)
class OptimisationResult:
    """Solution and derivative-sparsity diagnostics."""

    speeds: np.ndarray
    total_energy: float
    jacobian_nonzeros: int
    hessian_lower_nonzeros: int


def optimise_cruise_speeds() -> OptimisationResult:
    """Minimise energy with per-leg time constraints, using the lowered model.

    For drag = speed**2 + speed**-2, the exact optimum is
    max(1, required_speed) for each independent leg.
    """
    import asdex  # noqa: PLC0415
    import cyipopt  # noqa: PLC0415

    # Set x64 before generating arrays or compiling to match IPOPT precision.
    jax.config.update("jax_enable_x64", True)

    distances = jnp.asarray(DISTANCES)
    required = jnp.asarray(REQUIRED_SPEEDS)
    time_limits = distances / required
    n = len(DISTANCES)

    # All Bilt setup runs here, before any JAX tracing by grad or asdex.
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
    gradient_fn = jax.jit(jax.grad(objective))
    constraints_fn = jax.jit(constraints)

    # asdex analyses JAX primitives from the *lowered* calculation, not Bilt.
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
        """IPOPT host callbacks wrapping the already compiled JAX functions."""

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
            return np.asarray(
                hessian_fn(jnp.asarray(x), jnp.asarray(lagrange), jnp.asarray(obj_factor)).data
            )[lower]

    problem = cyipopt.Problem(
        n=n,
        m=n,
        problem_obj=Callbacks(),
        lb=np.full(n, 0.6),
        ub=np.full(n, 2.0),
        cl=np.full(n, -1e20),
        cu=np.zeros(n),
    )
    problem.add_option("tol", 1e-9)
    problem.add_option("print_level", 0)
    problem.add_option("sb", "yes")

    speeds, info = problem.solve(np.asarray(x0))
    if info["status"] not in (0, 1):
        msg = f"IPOPT did not converge: {info['status_msg']}"
        raise RuntimeError(msg)

    return OptimisationResult(
        speeds=np.asarray(speeds),
        total_energy=float(objective_fn(jnp.asarray(speeds))),
        jacobian_nonzeros=len(jac_rows),
        hessian_lower_nonzeros=len(hess_rows),
    )


if __name__ == "__main__":
    result = optimise_cruise_speeds()
    expected = np.maximum(1.0, np.asarray(REQUIRED_SPEEDS))
    np.testing.assert_allclose(result.speeds, expected, atol=1e-5, rtol=0)

"""Prototype: optimise independent cruise legs with Bilt, asdex and IPOPT.

The Bilt equations are evaluated by JAX and differentiated by asdex. IPOPT
receives sparse constraint Jacobians and Lagrangian Hessians through cyipopt.

All inputs are nondimensional for this first integration experiment. In
particular, JAX tracers are passed into scalar Bilt ports through jax.vmap,
rather than being converted to Bilt Tensor objects (which currently coerce
their values with NumPy).
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from bilt import Implementation, Interface, JAX_BACKEND, equation, field, system

# Independent route segments with differing arrival-time requirements.
DISTANCES = (1.0, 1.2, 0.8, 1.6, 1.1, 1.4, 0.9, 1.3)
REQUIRED_SPEEDS = (0.8, 1.15, 0.95, 1.3, 1.1, 0.9, 1.45, 1.0)


@system
class CruiseLeg(metaclass=Interface):
    """A deliberately simple, nondimensional aerodynamic mission leg."""

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


def evaluate_leg(speed: jax.Array, distance: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Evaluate the actual Bilt implementation, not a duplicated JAX formula."""
    with JAX_BACKEND, cruise_model:
        leg = CruiseLeg(
            speed=speed,
            distance=distance,
            parasite_coefficient=1.0,
            induced_coefficient=1.0,
        )
        return leg.energy, leg.time


def evaluate_legs(speeds: jax.Array, distances: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Map the scalar Bilt model over independent route legs."""
    return jax.vmap(evaluate_leg)(speeds, distances)


@dataclass(frozen=True)
class OptimisationResult:
    """Solution and derivative-sparsity diagnostics."""

    speeds: np.ndarray
    total_energy: float
    iterations: int
    jacobian_nonzeros: int
    hessian_lower_nonzeros: int


def optimise_cruise_speeds() -> OptimisationResult:
    """Minimise total energy with per-leg arrival-time constraints.

    The unconstrained optimum is speed=1 for every leg, because drag is
    speed**2 + speed**(-2). The time limits impose speed>=REQUIRED_SPEEDS,
    giving an independently checkable optimum max(1, required_speed).
    """
    import asdex
    import cyipopt

    # IPOPT needs double precision, and configuration must precede array creation.
    jax.config.update("jax_enable_x64", True)

    distances = jnp.asarray(DISTANCES, dtype=jnp.float64)
    required = jnp.asarray(REQUIRED_SPEEDS, dtype=jnp.float64)
    time_limits = distances / required
    n = len(DISTANCES)

    def objective(speeds: jax.Array) -> jax.Array:
        energies, _ = evaluate_legs(speeds, distances)
        return jnp.sum(energies)

    def constraints(speeds: jax.Array) -> jax.Array:
        _, times = evaluate_legs(speeds, distances)
        return times - time_limits  # <= 0

    def lagrangian(speeds: jax.Array, multipliers: jax.Array, factor: jax.Array) -> jax.Array:
        return factor * objective(speeds) + jnp.dot(multipliers, constraints(speeds))

    x0 = jnp.maximum(required, 1.2)
    objective_fn = jax.jit(objective)
    gradient_fn = jax.jit(jax.grad(objective))
    constraints_fn = jax.jit(constraints)

    # Sparsity detection/colouring happens once. Both derivative functions
    # return BCOO matrices with fixed coordinates and varying nonzero data.
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
        iterations=int(info.get("iter_count", -1)),
        jacobian_nonzeros=len(jac_rows),
        hessian_lower_nonzeros=len(hess_rows),
    )


if __name__ == "__main__":
    result = optimise_cruise_speeds()
    expected = np.maximum(1.0, np.asarray(REQUIRED_SPEEDS))
    print(f"IPOPT speeds:    {np.round(result.speeds, 6)}")
    print(f"Expected speeds: {expected}")
    print(f"Total energy:    {result.total_energy:.6f}")
    print(f"Jacobian nnz:    {result.jacobian_nonzeros}")
    print(f"Hessian nnz:     {result.hessian_lower_nonzeros}")
    np.testing.assert_allclose(result.speeds, expected, atol=1e-5, rtol=0)

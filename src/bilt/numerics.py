"""Numerical configuration for attribute-based model evaluation."""

from bilt._backend._nonlinear_solve import NonlinearControls as RootControls
from bilt._backend._nonlinear_solve import NonlinearResult, NonlinearSolver
from bilt._backend._python._nonlinear_solvers import ScipyHybrid, ScipyLeastSquares
from bilt.core._evaluation_options import EvaluationOptions, RootOptions, VariableOptions

__all__ = [
    "EvaluationOptions",
    "NonlinearResult",
    "NonlinearSolver",
    "RootControls",
    "RootOptions",
    "ScipyHybrid",
    "ScipyLeastSquares",
    "VariableOptions",
]

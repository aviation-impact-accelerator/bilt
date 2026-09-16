"""Causal and acausal systems models in Python."""

__all__ = [
    "JAX_BACKEND",
    "PYTHON_BACKEND",
    "BiltCoreError",
    "BiltNameError",
    "BiltSubSystemConflictError",
    "BiltSubSystemError",
    "BiltSubSystemMismatchedImplementationError",
    "BiltSubSystemMissingImplementationError",
    "BiltSubSystemNameCollisionError",
    "BiltSubSystemUnreferencedEntryError",
    "BiltSystemAnalysisError",
    "BiltSystemDefinitionError",
    "BiltUnknownSystemAttributeError",
    "CoordinateAxis",
    "Dim",
    "Domain",
    "EquationReturnError",
    "Implementation",
    "Interface",
    "Tensor",
    "as_domain",
    "coordinate_axis",
    "dim",
    "domain",
    "equation",
    "field",
    "system",
    "tensor",
    "transform",
]

from bilt._backend._jax import JAX_BACKEND
from bilt._backend._python import PYTHON_BACKEND
from bilt.core._equation import equation
from bilt.core._error import BiltCoreError, BiltNameError, BiltSystemAnalysisError, EquationReturnError
from bilt.core._interface import Interface
from bilt.core._system import (
    BiltSubSystemConflictError,
    BiltSubSystemError,
    BiltSubSystemMismatchedImplementationError,
    BiltSubSystemMissingImplementationError,
    BiltSubSystemNameCollisionError,
    BiltSubSystemUnreferencedEntryError,
    BiltSystemDefinitionError,
    BiltUnknownSystemAttributeError,
    Implementation,
    field,
    system,
)
from bilt.core._tensors._tensors import Tensor, tensor
from bilt.core._transform import transform
from bilt.symbolic._dimensions import CoordinateAxis, Dim, coordinate_axis, dim
from bilt.symbolic._domains import Domain, as_domain, domain

"""Bilt exception hierarchy.

Conventions:
1. Every bilt exception is defined in an `_errors.py` module beside the code that raises it, inherits `BiltError`
   through exactly one area sub-root, and is re-exported from `bilt.errors`. The exception is `BiltError` itself,
   which lives in `_base_error.py`.
2. Names use a `Bilt` prefix and `Error` suffix and name the fault, not the module that raises it.
3. Mix in a built-in exception only when the error replaces a built-in callers may already catch, or Python
   semantics expect it (`AttributeError` from attribute access, `TypeError` for wrong types, `ValueError` for bad
   values). The built-in comes second in the bases.
4. Every class has a docstring. Prefer structured attributes and a message built in `__init__` for new errors.
5. Internal invariants that indicate a bug in bilt raise built-in exceptions, not bilt errors.
6. Tests import errors only from `bilt.errors`.
"""

__all__ = [
    "BiltAnalysisError",
    "BiltAtomMutationError",
    "BiltBackendError",
    "BiltCompilationError",
    "BiltDefinitionError",
    "BiltDimensionalityError",
    "BiltDomainBindingError",
    "BiltDomainRegistryError",
    "BiltDomainRegistryLockedError",
    "BiltEquationReturnError",
    "BiltEquationSignatureError",
    "BiltError",
    "BiltEvaluationError",
    "BiltInvalidDomainError",
    "BiltIsolationError",
    "BiltMetadataError",
    "BiltNameError",
    "BiltNoActiveEvaluationError",
    "BiltPortDeclarationError",
    "BiltSubSystemConflictError",
    "BiltSubSystemError",
    "BiltSubSystemMismatchedImplementationError",
    "BiltSubSystemMissingImplementationError",
    "BiltSubSystemNameCollisionError",
    "BiltSubSystemUnreferencedEntryError",
    "BiltSymbolicConversionError",
    "BiltSymbolicError",
    "BiltSymbolicLoweringError",
    "BiltSystemDefinitionError",
    "BiltTargetNotFoundError",
    "BiltTargetOnBothSidesError",
    "BiltTensorDomainError",
    "BiltTreeMutationError",
    "BiltUnitError",
    "BiltUnitsError",
    "BiltUnknownPortError",
    "BiltUnknownSystemAttributeError",
    "BiltUnresolvedAnnotationError",
    "BiltUnsupportedIsolationError",
    "BiltUnsupportedNodeError",
]

from bilt._backend._errors import BiltBackendError, BiltCompilationError, BiltUnsupportedNodeError
from bilt._base_error import BiltError
from bilt.core._errors import (
    BiltAnalysisError,
    BiltDefinitionError,
    BiltDomainBindingError,
    BiltDomainRegistryError,
    BiltDomainRegistryLockedError,
    BiltEquationReturnError,
    BiltEquationSignatureError,
    BiltEvaluationError,
    BiltNameError,
    BiltNoActiveEvaluationError,
    BiltPortDeclarationError,
    BiltUnresolvedAnnotationError,
)
from bilt.core._system._errors import (
    BiltSubSystemConflictError,
    BiltSubSystemError,
    BiltSubSystemMismatchedImplementationError,
    BiltSubSystemMissingImplementationError,
    BiltSubSystemNameCollisionError,
    BiltSubSystemUnreferencedEntryError,
    BiltSystemDefinitionError,
    BiltUnknownSystemAttributeError,
)
from bilt.core._tensors._errors import BiltSymbolicLoweringError, BiltTensorDomainError
from bilt.symbolic._errors import (
    BiltAtomMutationError,
    BiltInvalidDomainError,
    BiltSymbolicConversionError,
    BiltSymbolicError,
    BiltTreeMutationError,
    BiltUnknownPortError,
)
from bilt.symbolic._isolation._errors import (
    BiltIsolationError,
    BiltTargetNotFoundError,
    BiltTargetOnBothSidesError,
    BiltUnsupportedIsolationError,
)
from bilt.units._errors import (
    BiltDimensionalityError,
    BiltMetadataError,
    BiltUnitError,
    BiltUnitsError,
)

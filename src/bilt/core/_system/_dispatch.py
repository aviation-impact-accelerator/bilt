__all__ = []

import typing

from bilt._backend._base_backend import _backend_stack
from bilt._backend._python import PYTHON_BACKEND
from bilt.core._system._errors import BiltUnknownSystemAttributeError
from bilt.core._structural_analysis._causalization_block_solver._errors import (
    BiltExecutionPlanError,
    BiltOverconstrainedError,
    BiltUnderconstrainedError,
)
from bilt.core._structural_analysis._causalization_block_solver._structural_validation import (
    validate_request_structure,
)

from ._sentinel import MISSING, _Missing

if typing.TYPE_CHECKING:
    from bilt.core._interface import Interface

    from ._decorator import Attribute
    from ._instance import SystemInstance


def _dispatch_evaluation[T](instance: SystemInstance, attribute: Attribute[T]) -> T:
    current = typing.cast("T | _Missing", instance.__bilt_system_instance_variables__.get(attribute.name, MISSING))
    if not isinstance(current, _Missing):
        return current
    implementation = instance.__bilt_system_implementation__
    cls = typing.cast("Interface", instance.__class__)
    if isinstance(implementation, _Missing):
        if len(cls.__bilt_system_implementation_default__) == 0:
            raise BiltUnknownSystemAttributeError(attribute.name, instance)
        implementation = cls.__bilt_system_implementation_default__[-1]
        instance.__bilt_system_implementation__ = implementation
    fixed_inputs = {name: value for name, value in instance.__bilt_system_instance_inputs__.items() if value is not MISSING}
    cached_inputs = {name: value for name, value in instance.__bilt_system_instance_variables__.items() if value is not MISSING}
    backend = _backend_stack[-1] if _backend_stack else PYTHON_BACKEND
    output = attribute.name
    if cached_inputs.keys() - fixed_inputs.keys():
        # Cached computations are not independent structural constraints.
        validate_request_structure(implementation.structural_analysis().incidence, set(fixed_inputs), output)
        try:
            outputs = implementation._evaluate_with_intermediates(  # noqa: SLF001
                inputs=cached_inputs, output=output, backend=backend
            )
        except (BiltExecutionPlanError, BiltOverconstrainedError, BiltUnderconstrainedError):
            outputs = implementation._evaluate_with_intermediates(  # noqa: SLF001
                inputs=fixed_inputs, output=output, backend=backend
            )
    else:
        outputs = implementation._evaluate_with_intermediates(  # noqa: SLF001
            inputs=fixed_inputs, output=output, backend=backend
        )
    if output not in outputs:
        msg = f"Evaluating {output!r} did not produce the requested value."
        raise RuntimeError(msg)
    # Results are published atomically, without replacing any already known values.
    new_outputs = {name: value for name, value in outputs.items() if name not in cached_inputs}
    instance.__bilt_system_instance_variables__.update(new_outputs)
    return typing.cast("T", outputs[output])

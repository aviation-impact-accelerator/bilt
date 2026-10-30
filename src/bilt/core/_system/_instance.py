__all__ = []

import typing

if typing.TYPE_CHECKING:
    from bilt._backend import Backend
    from bilt.core._evaluation_options import EvaluationOptions
    from bilt.core._structural_analysis._nonlinear_cache import NonlinearSolutionKey

    from ._implementation import Implementation
    from ._sentinel import _Missing


class SystemInstance(typing.Protocol):
    __bilt_system_instance_variables__: dict[str, object]
    __bilt_system_instance_inputs__: dict[str, object]
    __bilt_system_instance_evaluation_cache__: dict[tuple[object, EvaluationOptions | None, Backend], dict[str, object]]
    __bilt_system_instance_nonlinear_cache__: dict[
        tuple[object, EvaluationOptions | None, Backend], dict[NonlinearSolutionKey, dict[str, float]]
    ]
    __bilt_system_implementation__: Implementation | _Missing

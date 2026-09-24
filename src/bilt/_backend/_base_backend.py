"""Minimal interface for bilt backend, that will be utilised by the analysis layer."""

import typing

if typing.TYPE_CHECKING:
    import ast

    from bilt.core._tensors._tensor_alignment import AlignmentPlan

    from ._callable import BackendCallable
    from ._case import BackendCase

_backend_stack: list[Backend] = []


class Backend(typing.Protocol):
    """Interface a backend must expose. `ActivatableBackendMixin` supplies the context manager half."""

    def transpile(self, function_definition: ast.FunctionDef) -> BackendCallable:
        """Compile a single transform AST into a callable function, that can then be chained into the execution pipeline."""
        ...

    def chain_case(self, case: BackendCase) -> BackendCallable:
        """Chain the callables of a case into a single callable.

        If the case has expose_intermediates is true: returns a tuple of all outputs, in the order
        `case.output_names` lists them.
        If the case has expose_intermediates is false: returns a single object corresponding to return name.
        """
        ...

    def __enter__(self) -> typing.Self: ...

    def __exit__(self, *_: object) -> None: ...

    def apply_alignment(self, lowered_inputs: dict[str, object], alignment_plan: AlignmentPlan) -> dict[str, object]:
        """Apply the given alignment plan on the provided lowered inputs."""


class ActivatableBackendMixin:
    """Makes a backend the one dispatch selects for the duration of a `with` block."""

    def __enter__[T: Backend](self: T) -> T:  # noqa: PYI019
        _backend_stack.append(self)
        return self

    def __exit__(self, *_: object) -> None:
        _backend_stack.pop()

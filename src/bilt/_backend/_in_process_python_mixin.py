"""Behaviour shared by backends that run the transform AST in this Python process."""

__all__ = ["InProcessPythonMixin", "PythonIntrinsicBindings"]

import ast
import dataclasses
import typing

from bilt._backend._callable import BackendCallable
from bilt._backend._errors import BiltCompilationError, BiltUnsupportedNodeError
from bilt._backend._linear_solve import (
    LINEAR_BASIS_INTRINSIC_NAME,
    LINEAR_EXPAND_INTRINSIC_NAME,
    LINEAR_SOLVE_INTRINSIC_NAME,
    LINEAR_ZERO_INTRINSIC_NAME,
)
from bilt.core._to_ast import (
    ISEL_INTRINSIC_NAME,
    MAX_INTRINSIC_NAME,
    MEAN_INTRINSIC_NAME,
    MIN_INTRINSIC_NAME,
    PROD_INTRINSIC_NAME,
    SUM_INTRINSIC_NAME,
)


@dataclasses.dataclass(frozen=True, slots=True)
class PythonIntrinsicBindings:
    """Backend implementations of Bilt intrinsics."""

    isel: typing.Callable[..., typing.Any]
    mean: typing.Callable[..., typing.Any]
    sum: typing.Callable[..., typing.Any]
    min: typing.Callable[..., typing.Any]
    max: typing.Callable[..., typing.Any]
    prod: typing.Callable[..., typing.Any]
    linear_solve: typing.Callable[..., typing.Any] | None = None

    linear_basis: typing.Callable[..., typing.Any] | None = None
    linear_expand: typing.Callable[..., typing.Any] | None = None
    linear_zero: typing.Callable[..., typing.Any] | None = None

    def namespace(self) -> dict[str, typing.Any]:
        namespace = {
            ISEL_INTRINSIC_NAME: self.isel,
            MEAN_INTRINSIC_NAME: self.mean,
            SUM_INTRINSIC_NAME: self.sum,
            MIN_INTRINSIC_NAME: self.min,
            MAX_INTRINSIC_NAME: self.max,
            PROD_INTRINSIC_NAME: self.prod,
        }
        if self.linear_solve is not None:
            namespace[LINEAR_SOLVE_INTRINSIC_NAME] = self.linear_solve
        if self.linear_basis is not None:
            namespace[LINEAR_BASIS_INTRINSIC_NAME] = self.linear_basis
        if self.linear_expand is not None:
            namespace[LINEAR_EXPAND_INTRINSIC_NAME] = self.linear_expand
        if self.linear_zero is not None:
            namespace[LINEAR_ZERO_INTRINSIC_NAME] = self.linear_zero
        return namespace


class InProcessPythonMixin:
    """Rewrites the transform AST, then compiles and executes it on the host interpreter rather than out of process."""

    ast_transformer: ast.NodeTransformer
    compilation_namespace: dict[str, typing.Any]
    intrinsic_bindings: PythonIntrinsicBindings

    def transpile_linear_solve(self, function_definition: ast.FunctionDef) -> BackendCallable:
        """Compile a solve using this backend's intrinsic and compilation policy."""
        if self.intrinsic_bindings.linear_solve is None:
            message = "The selected backend does not support linear blocks."
            raise BiltUnsupportedNodeError(message)
        namespace = self.intrinsic_bindings.namespace()
        for node in ast.walk(function_definition):
            if (
                isinstance(node, ast.Name)
                and node.id in (LINEAR_BASIS_INTRINSIC_NAME, LINEAR_EXPAND_INTRINSIC_NAME, LINEAR_ZERO_INTRINSIC_NAME)
                and node.id not in namespace
            ):
                message = "The selected backend does not support joint tensor linear blocks."
                raise BiltUnsupportedNodeError(message)
        # The enclosing case is compiled as a whole. Avoid an additional
        # zero-argument JIT boundary around constant systems.
        function_definition.__dict__["_bilt_inline"] = True
        return self.transpile(function_definition)

    def transpile(self, function_definition: ast.FunctionDef) -> BackendCallable:
        """Rewrite the definition with `ast_transformer`, then compile and exec it into `compilation_namespace`."""
        transformed = self.ast_transformer.visit(function_definition)
        module = ast.Module(body=[transformed], type_ignores=[])
        ast.fix_missing_locations(module)

        namespace: dict[str, typing.Any] = {
            **self.compilation_namespace,
            **typing.cast("dict[str, object]", getattr(function_definition, "_bilt_runtime_bindings", {})),
            **self.intrinsic_bindings.namespace(),
        }

        try:
            code = compile(module, "<backend_compile>", "exec")

        except Exception as e:
            msg = f"Error when trying to compile transformed function '{function_definition.name}' threw error: {e}"
            raise BiltCompilationError(msg) from e

        exec(code, namespace)  # noqa: S102
        return BackendCallable(
            name=function_definition.name,
            parameters=tuple(arg.arg for arg in function_definition.args.args),
            callable=namespace[function_definition.name],
        )

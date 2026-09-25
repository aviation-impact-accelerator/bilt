"""Behaviour shared by backends that run the transform AST in this Python process."""

__all__ = ["InProcessPythonMixin", "PythonIntrinsicBindings"]

import ast
import dataclasses
import typing

from bilt._backend._callable import BackendCallable
from bilt._backend._errors import BiltCompilationError
from bilt.core._to_ast import MEAN_INTRINSIC_NAME


@dataclasses.dataclass(frozen=True, slots=True)
class PythonIntrinsicBindings:
    """Backend implementations of Bilt intrinsics."""

    mean: typing.Callable[..., typing.Any]

    def namespace(self) -> dict[str, typing.Any]:
        return {
            MEAN_INTRINSIC_NAME: self.mean,
        }


class InProcessPythonMixin:
    """Rewrites the transform AST, then compiles and executes it on the host interpreter rather than out of process."""

    ast_transformer: ast.NodeTransformer
    compilation_namespace: dict[str, typing.Any]
    intrinsic_bindings: PythonIntrinsicBindings

    def transpile(self, function_definition: ast.FunctionDef) -> BackendCallable:
        """Rewrite the definition with `ast_transformer`, then compile and exec it into `compilation_namespace`."""
        transformed = self.ast_transformer.visit(function_definition)
        module = ast.Module(body=[transformed], type_ignores=[])
        ast.fix_missing_locations(module)

        namespace = {
            **self.compilation_namespace,
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

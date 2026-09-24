"""Behaviour shared by backends that run the transform AST in this Python process."""

__all__ = ["InProcessPythonMixin"]

import ast
import typing

from bilt.backend._callable import BackendCallable
from bilt.backend._errors import BiltCompilationError


class InProcessPythonMixin:
    """Rewrites the transform AST, then compiles and executes it on the host interpreter rather than out of process."""

    ast_transformer: ast.NodeTransformer
    compilation_namespace: dict[str, typing.Any]

    def transpile(self, function_definition: ast.FunctionDef) -> BackendCallable:
        """Rewrite the definition with `ast_transformer`, then compile and exec it into `compilation_namespace`."""
        transformed = self.ast_transformer.visit(function_definition)
        module = ast.Module(body=[transformed], type_ignores=[])
        ast.fix_missing_locations(module)
        try:
            code = compile(module, "<backend_compile>", "exec")
        except Exception as e:
            msg = f"Error when trying to compile transformed function '{function_definition.name}' threw error: {e}"
            raise BiltCompilationError(msg) from e

        exec(code, self.compilation_namespace)  # noqa: S102
        return BackendCallable(
            name=function_definition.name,
            parameters=tuple(arg.arg for arg in function_definition.args.args),
            callable=self.compilation_namespace[function_definition.name],
        )

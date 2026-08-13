"""Behaviour shared by backends that interpret the transform AST as Python."""

__all__ = ["PythonInterpretable"]

import ast
import typing

from bilt._backend._callable import BackendCallable
from bilt._backend._errors import CompilationError


class PythonInterpretable:
    """Mixin for backends that rewrite the transform AST, then compile and execute it as Python."""

    def __init__(self, ast_transformer: ast.NodeTransformer, compilation_namespace: dict[str, typing.Any]) -> None:
        self.ast_transformer = ast_transformer
        self.compilation_namespace = compilation_namespace

    def transpile(self, function_definition: ast.FunctionDef) -> BackendCallable:
        """Rewrite the definition with `ast_transformer`, then compile and exec it into `compilation_namespace`."""
        transformed = self.ast_transformer.visit(function_definition)
        module = ast.Module(body=[transformed], type_ignores=[])
        ast.fix_missing_locations(module)
        try:
            code = compile(module, "<backend_compile>", "exec")
        except Exception as e:
            msg = f"Error when trying to compile transformed function '{function_definition.name}' threw error: {e}"
            raise CompilationError(msg) from e

        exec(code, self.compilation_namespace)  # noqa: S102
        return BackendCallable(
            name=function_definition.name,
            parameters=tuple(arg.arg for arg in function_definition.args.args),
            callable=self.compilation_namespace[function_definition.name],
        )

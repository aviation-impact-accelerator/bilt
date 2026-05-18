"""Tiny Python-equation DSL compiler for pure Python and JAX backends.

This intentionally supports only scalar equation functions for now.
It does not lower loops, comprehensions, mutation, classes, or control flow.

The goal is:
    - write equations as normal Python functions using math.exp/log/pi
    - keep those equations executable in pure Python
    - compile equivalent JAX functions by rewriting the AST
"""

import ast
import inspect
import textwrap
from collections.abc import Callable
from dataclasses import dataclass
from types import FunctionType
from typing import Any, Literal

BackendName = Literal["python", "jax"]


class UnsupportedDslError(TypeError):
    """Raised when a function uses Python syntax outside the supported DSL."""


class EquationDslValidator(ast.NodeVisitor):
    """Validate the deliberately small scalar-equation DSL.

    Allowed in practice:
        - function definitions
        - assignment
        - return
        - arithmetic expressions
        - function calls
        - tuples
        - names/constants

    Rejected for now:
        - loops
        - if statements
        - comprehensions
        - try/with/class/lambda/yield/async/global/nonlocal
    """

    unsupported_nodes = (
        ast.For,
        ast.While,
        ast.If,
        ast.With,
        ast.Try,
        ast.ListComp,
        ast.DictComp,
        ast.SetComp,
        ast.GeneratorExp,
        ast.Lambda,
        ast.Yield,
        ast.YieldFrom,
        ast.Await,
        ast.AsyncFunctionDef,
        ast.ClassDef,
        ast.Delete,
        ast.Global,
        ast.Nonlocal,
    )

    def generic_visit(self, node: ast.AST) -> None:
        if isinstance(node, self.unsupported_nodes):
            raise UnsupportedDslError(f"Unsupported DSL node: {type(node).__name__}")
        super().generic_visit(node)


class DecoratorStripper(ast.NodeTransformer):
    """Remove decorators from functions before recompiling them.

    This allows a source function written as:

        @equation
        def f(...): ...

    to compile back to a normal callable named f.
    """

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        node.decorator_list = []
        self.generic_visit(node)
        return node


class MathToJaxTransformer(ast.NodeTransformer):
    """Rewrite math-like scalar equations to JAX-compatible equations.

    Examples:
        exp(x)       -> jnp.exp(x)
        log(x)       -> jnp.log(x)
        sqrt(x)      -> jnp.sqrt(x)
        abs(x)       -> jnp.abs(x)
        pi           -> jnp.pi
        e            -> jnp.e
        math.exp(x)  -> jnp.exp(x)
        math.pi      -> jnp.pi
    """

    math_functions = {
        "exp",
        "log",
        "sqrt",
        "sin",
        "cos",
        "tan",
        "abs",
    }

    math_constants = {
        "pi",
        "e",
    }

    def visit_ImportFrom(self, node: ast.ImportFrom) -> ast.AST | None:
        # Remove: from math import exp, log, pi
        if node.module == "math":
            return None
        return node

    def visit_Import(self, node: ast.Import) -> ast.AST | None:
        # Remove: import math, but preserve unrelated imports.
        kept = [alias for alias in node.names if alias.name != "math"]
        if not kept:
            return None
        node.names = kept
        return node

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if isinstance(node.ctx, ast.Load) and node.id in self.math_constants:
            return ast.copy_location(
                ast.Attribute(
                    value=ast.Name(id="jnp", ctx=ast.Load()),
                    attr=node.id,
                    ctx=ast.Load(),
                ),
                node,
            )
        return node

    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)

        # exp(x) -> jnp.exp(x)
        if isinstance(node.func, ast.Name) and node.func.id in self.math_functions:
            node.func = ast.Attribute(
                value=ast.Name(id="jnp", ctx=ast.Load()),
                attr=node.func.id,
                ctx=ast.Load(),
            )
            return node

        # math.exp(x) -> jnp.exp(x)
        if isinstance(node.func, ast.Attribute):
            if (
                isinstance(node.func.value, ast.Name)
                and node.func.value.id == "math"
                and node.func.attr in self.math_functions
            ):
                node.func = ast.Attribute(
                    value=ast.Name(id="jnp", ctx=ast.Load()),
                    attr=node.func.attr,
                    ctx=ast.Load(),
                )
                return node

        return node

    def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
        self.generic_visit(node)

        # math.pi -> jnp.pi
        if isinstance(node.value, ast.Name) and node.value.id == "math" and node.attr in self.math_constants:
            return ast.copy_location(
                ast.Attribute(
                    value=ast.Name(id="jnp", ctx=ast.Load()),
                    attr=node.attr,
                    ctx=ast.Load(),
                ),
                node,
            )

        return node


@dataclass(frozen=True)
class Equation:
    """A Python function treated as a backend-compilable scalar equation."""

    fn: Callable[..., Any]

    @property
    def name(self) -> str:
        return self.fn.__name__

    @property
    def source(self) -> str:
        return textwrap.dedent(inspect.getsource(self.fn))

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Allow decorated equations to call each other in pure Python mode."""
        return self.fn(*args, **kwargs)

    def compile(
        self,
        backend: BackendName,
        *,
        extra_namespace: dict[str, Any] | None = None,
    ) -> Callable[..., Any]:
        if backend == "python":
            return self.fn
        if backend == "jax":
            return compile_equation_to_jax(
                source=self.source,
                function_name=self.name,
                extra_namespace=extra_namespace,
            )
        raise ValueError(f"Unknown backend: {backend!r}")


def equation(fn: Callable[..., Any]) -> Equation:
    """Decorate a scalar function as a backend-compilable equation."""
    return Equation(fn)


def compile_equation_to_jax(
    source: str,
    function_name: str,
    *,
    extra_namespace: dict[str, Any] | None = None,
) -> Callable[..., Any]:
    """Compile one scalar equation function from source into a JAX-compatible function."""
    import jax.numpy as jnp

    tree = ast.parse(textwrap.dedent(source))

    EquationDslValidator().visit(tree)
    tree = DecoratorStripper().visit(tree)
    tree = MathToJaxTransformer().visit(tree)
    ast.fix_missing_locations(tree)

    namespace: dict[str, Any] = {"jnp": jnp}
    if extra_namespace:
        namespace.update(extra_namespace)

    code = compile(tree, filename="<equation-dsl>", mode="exec")
    exec(code, namespace)

    compiled = namespace[function_name]
    if not isinstance(compiled, FunctionType):
        raise TypeError(f"{function_name!r} did not compile to a function")

    return compiled

import ast


class JaxASTTransformer(ast.NodeTransformer):
    def visit_List(self, node: ast.List) -> ast.Call:
        self.generic_visit(node)
        return ast.Call(
            func=ast.Attribute(
                value=ast.Name(id="jnp", ctx=ast.Load()),
                attr="array",
                ctx=ast.Load(),
            ),
            args=[node],
            keywords=[],
        )

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        self.generic_visit(node)
        node.decorator_list.insert(0, ast.Name(id="jax.jit", ctx=ast.Load()))
        return node

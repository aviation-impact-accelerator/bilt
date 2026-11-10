import ast


class JAXASTTransformer(ast.NodeTransformer):
    def visit_IfExp(self, node: ast.IfExp) -> ast.Call:
        self.generic_visit(node)
        argument = ast.arg(arg="_", annotation=None)

        def branch(value: ast.expr) -> ast.Lambda:
            return ast.Lambda(
                args=ast.arguments(
                    posonlyargs=[],
                    args=[argument],
                    kwonlyargs=[],
                    kw_defaults=[],
                    defaults=[],
                    vararg=None,
                    kwarg=None,
                ),
                body=value,
            )

        return ast.copy_location(
            ast.Call(
                func=ast.Name(id="jax.lax.cond", ctx=ast.Load()),
                args=[
                    node.test,
                    branch(node.body),
                    branch(node.orelse),
                    ast.Constant(value=None),
                ],
                keywords=[],
            ),
            node,
        )

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
        if getattr(node, "_bilt_inline", False):
            return node
        node.decorator_list.insert(0, ast.Name(id="jax.jit", ctx=ast.Load()))
        return node

"""Conservative lexical source analysis shared by Bilt tooling."""

import ast
import builtins
import symtable
import typing
from contextlib import contextmanager

if typing.TYPE_CHECKING:
    from collections.abc import Generator


def _import_names(statement: ast.Import | ast.ImportFrom) -> dict[str, str]:
    if isinstance(statement, ast.Import):
        return {
            alias.asname or alias.name.split(".")[0]: alias.name if alias.asname else alias.name.split(".")[0]
            for alias in statement.names
        }
    if statement.level or statement.module is None:
        return {}
    return {
        alias.asname or alias.name: f"{statement.module}.{alias.name}" for alias in statement.names if alias.name != "*"
    }


def _stable_imports(statements: list[ast.stmt], table: symtable.SymbolTable, node: ast.AST) -> set[str]:
    """Use only unconditional, uniquely bound imports; never infer control flow."""
    candidates = {
        name
        for statement in statements
        if isinstance(statement, (ast.Import, ast.ImportFrom))
        for name in _import_names(statement)
    }
    counts: dict[str, int] = {}
    for item in ast.walk(node):
        if isinstance(item, (ast.Import, ast.ImportFrom)):
            for name in _import_names(item):
                counts[name] = counts.get(name, 0) + 1
    return {
        name
        for name in candidates.intersection(table.get_identifiers())
        if counts[name] == 1
        and table.lookup(name).is_imported()
        and not table.lookup(name).is_assigned()
        and not table.lookup(name).is_namespace()
    }


class LexicalScopes:
    """Conservative lexical bindings for source analysis, independent of runtime values."""

    def __init__(self, source: str, tree: ast.Module) -> None:
        root = symtable.symtable(source, "<source>", "exec")
        self.functions: dict[tuple[str, int], list[symtable.SymbolTable]] = {}
        self.names = set(root.get_identifiers())
        self._index(root)
        globals_written = {name for node in ast.walk(tree) if isinstance(node, ast.Global) for name in node.names}
        stable = _stable_imports(tree.body, root, tree) - globals_written
        self.builtin_names = (
            set(dir(builtins))
            - globals_written
            - {name for name in root.get_identifiers() if root.lookup(name).is_local()}
        )
        if any(
            isinstance(node, ast.ImportFrom) and any(alias.name == "*" for alias in node.names)
            for node in ast.walk(tree)
        ):
            stable.clear()
            self.builtin_names.clear()
        imports: dict[str, str] = {}
        self.lines: dict[str, int] = {}
        for statement in tree.body:
            if isinstance(statement, (ast.Import, ast.ImportFrom)):
                imports.update(_import_names(statement))
                for name in _import_names(statement):
                    self.lines[name] = statement.lineno
        self.bindings = {name: value for name, value in imports.items() if name in stable}

    def _index(self, table: symtable.SymbolTable) -> None:
        for child in table.get_children():
            self.names.update(child.get_identifiers())
            if child.get_type() == symtable.SymbolTableType.FUNCTION:
                # Python versions may expose the compiler name "<lambda>".
                name = child.get_name()
                key = ("lambda" if name == "<lambda>" else name, child.get_lineno())
                self.functions.setdefault(key, []).append(child)
            self._index(child)

    def function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> symtable.SymbolTable | None:
        tables = self.functions.get((node.name, node.lineno), [])
        return tables[0] if len(tables) == 1 else None

    def globals_for(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda, table: symtable.SymbolTable
    ) -> set[str]:
        return {
            name
            for name in self.bindings
            if self.lines[name] < node.lineno
            and (name not in table.get_identifiers() or table.lookup(name).is_global())
        }


class LexicalVisitor(ast.NodeVisitor):
    """Traverse lexical scopes without importing or executing the source."""

    def __init__(self, source: str, tree: ast.Module) -> None:
        self.bindings: dict[str, str | None] = {}
        self.tree = tree
        self.lexical = LexicalScopes(source, tree)
        self.allowed: set[str] | None = None
        self.in_class = False
        self.parameter_members: dict[str, set[str]] = {}
        self.builtin_names = self.lexical.builtin_names
        self.names = self.lexical.names

    def unresolved_scope(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        """Hook for consumers that need to report an ambiguous scope."""

    def builtin_visible(self, name: str) -> bool:
        """Return whether a builtin is unshadowed in the current scope."""
        return name in self.builtin_names

    @contextmanager
    def _scope(self) -> Generator[None]:
        outer = self.bindings
        self.bindings = outer.copy()
        try:
            yield
        finally:
            self.bindings = outer

    def resolve(self, node: ast.expr | None) -> str | None:
        """Resolve imported names and straightforward reference aliases, not inferred value types."""
        if isinstance(node, ast.Name):
            if self.allowed is not None and node.id not in self.allowed:
                return None
            return self.bindings.get(node.id)
        if isinstance(node, ast.Attribute) and (parent := self.resolve(node.value)) is not None:
            return f"{parent}.{node.attr}"
        return None

    @typing.override
    def visit_Import(self, node: ast.Import) -> None:
        self.bindings.update(_import_names(node))

    @typing.override
    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        self.bindings.update(_import_names(node))

    @typing.override
    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, (ast.Store, ast.Del)):
            self.bindings[node.id] = None

    @typing.override
    def visit_Assign(self, node: ast.Assign) -> None:
        self.visit(node.value)
        reference = self.resolve(node.value)
        for target in node.targets:
            self.visit(target)
            if isinstance(target, ast.Name):
                self.bindings[target.id] = reference

    @typing.override
    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        reference = self.resolve(node.value)
        self.generic_visit(node)
        if isinstance(node.target, ast.Name):
            self.bindings[node.target.id] = reference

    def union_members(self, node: ast.expr | None) -> set[str]:
        """Recognise direct union members, not types nested in containers."""
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
            return self.union_members(node.left) | self.union_members(node.right)
        if (
            isinstance(node, ast.Subscript)
            and self.resolve(node.value) == "typing.Union"
            and isinstance(node.slice, ast.Tuple)
        ):
            return set().union(*(self.union_members(item) for item in node.slice.elts))
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            try:
                return self.union_members(ast.parse(node.value, mode="eval").body)
            except SyntaxError:
                return set()
        kind = self.resolve(node)
        return {kind} if kind is not None else set()

    def _parameter_members(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef, table: symtable.SymbolTable
    ) -> dict[str, set[str]]:
        """Record direct union members of parameters without local reassignment."""
        nonlocals = {name for item in ast.walk(node) if isinstance(item, ast.Nonlocal) for name in item.names}
        arguments = (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs)
        return {
            argument.arg: self.union_members(argument.annotation)
            for argument in arguments
            if argument.arg in table.get_identifiers()
            and not table.lookup(argument.arg).is_assigned()
            and argument.arg not in nonlocals
        }

    def _visible_builtins(self, table: symtable.SymbolTable) -> set[str]:
        return {
            name
            for name in self.lexical.builtin_names
            if name not in table.get_identifiers() or table.lookup(name).is_global()
        }

    @typing.override
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._function(node)

    @typing.override
    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._function(node)

    def _function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        # Defaults, decorators and annotations belong to the enclosing scope.
        for decorator in node.decorator_list:
            self.visit(decorator)
        self.visit(node.args)
        if node.returns is not None:
            self.visit(node.returns)
        self.bindings[node.name] = None

        table = self.lexical.function(node)
        if table is None:
            self.unresolved_scope(node)
            return
        parameters = self._parameter_members(node, table)
        globals_visible = self.lexical.globals_for(node, table)
        local_imports = _stable_imports(node.body, table, node)
        outer_allowed = self.allowed
        outer_class = self.in_class
        outer_members = self.parameter_members
        outer_builtin = self.builtin_names
        with self._scope():
            self.bindings = {name: self.lexical.bindings[name] for name in globals_visible}
            self.allowed = globals_visible | local_imports
            self.in_class = False
            self.parameter_members = parameters
            self.builtin_names = self._visible_builtins(table)
            try:
                for statement in node.body:
                    self.visit(statement)
            finally:
                self.allowed = outer_allowed
                self.in_class = outer_class
                self.parameter_members = outer_members
                self.builtin_names = outer_builtin

    @typing.override
    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        for expression in (*node.decorator_list, *node.bases, *node.keywords):
            self.visit(expression)
        self.bindings[node.name] = None
        outer_class = self.in_class
        outer_members = self.parameter_members
        outer_builtin = self.builtin_names
        with self._scope():
            self.in_class = True
            self.parameter_members = {}
            self.builtin_names = set()
            try:
                for statement in node.body:
                    self.visit(statement)
            finally:
                self.in_class = outer_class
                self.parameter_members = outer_members
                self.builtin_names = outer_builtin

    @typing.override
    def visit_Lambda(self, node: ast.Lambda) -> None:
        self.visit(node.args)
        tables = self.lexical.functions.get(("lambda", node.lineno), [])
        if len(tables) != 1:
            return
        table = tables[0]
        visible = self.lexical.globals_for(node, table)
        outer_allowed = self.allowed
        outer_class = self.in_class
        outer_members = self.parameter_members
        outer_builtin = self.builtin_names
        with self._scope():
            self.bindings = {name: self.lexical.bindings[name] for name in visible}
            self.allowed = visible
            self.in_class = False
            self.parameter_members = {}
            self.builtin_names = self._visible_builtins(table)
            try:
                self.visit(node.body)
            finally:
                self.allowed = outer_allowed
                self.in_class = outer_class
                self.parameter_members = outer_members
                self.builtin_names = outer_builtin

    def _comprehension(self, node: ast.ListComp | ast.SetComp | ast.DictComp | ast.GeneratorExp) -> None:
        # The first iterable is evaluated in the surrounding scope; every target
        # is local to the comprehension body, including later generator targets.
        first, *rest = node.generators
        self.visit(first.iter)
        if self.in_class:
            # Class suites are not closure scopes for comprehension bodies.
            return
        blocked = {
            item.id
            for generator in node.generators
            for item in ast.walk(generator.target)
            if isinstance(item, ast.Name) and isinstance(item.ctx, ast.Store)
        }
        blocked.update(
            item.target.id
            for item in ast.walk(node)
            if isinstance(item, ast.NamedExpr) and isinstance(item.target, ast.Name)
        )
        outer_allowed = self.allowed
        outer_members = self.parameter_members
        outer_builtin = self.builtin_names
        allowed = set(self.lexical.bindings) if outer_allowed is None else outer_allowed
        with self._scope():
            self.allowed = allowed - blocked
            self.parameter_members = {name: members for name, members in outer_members.items() if name not in blocked}
            self.builtin_names = outer_builtin - blocked
            try:
                for condition in first.ifs:
                    self.visit(condition)
                for generator in rest:
                    self.visit(generator.iter)
                    for condition in generator.ifs:
                        self.visit(condition)
                if isinstance(node, ast.DictComp):
                    self.visit(node.key)
                    if node.value is not None:
                        self.visit(node.value)
                else:
                    self.visit(node.elt)
            finally:
                self.allowed = outer_allowed
                self.parameter_members = outer_members
                self.builtin_names = outer_builtin
        for item in ast.walk(node):
            if isinstance(item, ast.NamedExpr) and isinstance(item.target, ast.Name):
                self.bindings[item.target.id] = None

    @typing.override
    def visit_ListComp(self, node: ast.ListComp) -> None:
        self._comprehension(node)

    @typing.override
    def visit_SetComp(self, node: ast.SetComp) -> None:
        self._comprehension(node)

    @typing.override
    def visit_DictComp(self, node: ast.DictComp) -> None:
        self._comprehension(node)

    @typing.override
    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:
        self._comprehension(node)

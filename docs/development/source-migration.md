# Source migration framework

Bilt's internal migration engine is independent of individual API changes. A migration
consists of an ordered tuple of rules, each inspecting an expression in its lexical
context and optionally proposing a replacement or a manual-review diagnostic.

Rules implement `MigrationRule.apply(node, context)` in `bilt._migration`.
The engine supplies source spans, import and alias resolution, nested edit composition,
collision-free imports, dry runs, file discovery and write mode. It does not import
or execute the source being migrated. Rewrites are validated with the Python parser.

The framework is internal; the user-facing command and concrete migration rules are
introduced with the API change that needs them. Tests exercise the engine using
an illustrative non-Bilt API rename.

Scope resolution is deliberately conservative and is not a complete type checker.
Rules should require explicit evidence before rewriting expressions and leave
ambiguous changes for review. Preserve comments and evaluation order, and test
idempotence and shadowing for each rule.

## Optional source cleanup

A migration may register a `CleanupOptions` policy and pass it to
`migrate_source` or `migrate`. Explicitly list fully qualified imports that
are safe to remove. Only imports whose last apparent reference disappeared
during the migration are eligible. Pre-existing unused imports, explicit
re-exports, commented imports, quoted references, package initialisers and
modules using dynamic namespace access are conservatively retained. This is
not a whole-project unused-import analysis.

A caller may also supply a source formatter. The engine rejects formatting
that changes the Python AST. Neither an external formatter nor additional
dependencies are required by the framework. Cleanup runs only on changed
source with no outstanding migration notes.

## Expression generation

Use `context.replace_binary(node, left, "*", right)` to construct a binary
operation from two source expressions. The helper composes nested edits and
checks candidate expressions against a fully grouped reference AST. It removes
parentheses only if the operand grouping and enclosing Python syntax are
unchanged. This preserves evaluation order even for overloaded arithmetic and
non-associative operators.

For other already-rendered expressions, `context.replace_expression(node,
replacement)` applies the same contextual grouping check. Both helpers retain
the engine's existing comment protection and manual-review diagnostics. No
source is executed and no formatter dependency is required.


## Advisory review

An optional `review(original, migrated)` callback runs after rewriting and
cleanup, even when no edits were needed. It returns source-located notes without
changing source. Unlike notes raised by an unsuccessful rewrite, these findings
do not suppress safe cleanup or formatting. Review locations refer to the
candidate migrated source (the output in preview mode). The CLI returns a nonzero
status when review items remain, including in write mode.

The quantity migration uses this pass to flag changed `Quantity` runtime checks,
negative `QuantityTensor` checks where scalar narrowing may be assumed, magnitude
access on a recognisable scalar-unit construction, and newly unused approved
imports retained by cleanup. Review these according to the intended shape;
do not automatically insert casts, scalar conversions or reductions. Retained
import findings describe the current rewrite, not pre-existing unused imports.
These syntactic checks are intentionally incomplete: run the project's type
checker, linter and regression tests after migration. A zero review count is
not proof of semantic equivalence.

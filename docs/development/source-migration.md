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

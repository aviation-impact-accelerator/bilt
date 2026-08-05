# Graph Report - /Users/je484/Code/bilt  (2026-08-05)

## Corpus Check
- cluster-only mode — file stats not available

## Summary
- 957 nodes · 2451 edges · 49 communities (42 shown, 7 thin omitted)
- Extraction: 91% EXTRACTED · 9% INFERRED · 0% AMBIGUOUS · INFERRED: 215 edges (avg confidence: 0.55)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `db0c8d90`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- Domain
- Tree
- Quantity
- _units.py
- test_atoms.py
- Unit
- .symbol
- symbol_tree
- Dimension
- test_system.py
- UnitOrder
- _isolate.py
- DomainRegistry
- UnitComposition
- Atom
- _expr_ops.py
- core/__init__.py
- JAXBackend
- _exhaustive/_backend.py
- Metadata
- Interface
- symbolic/_dimensions.py
- _to_ast.py
- test_implementation.py
- Tensor
- Backend
- BackendCallable
- _tensors.py
- PythonBackend
- test_equation.py
- _tube_geometry.py
- _supports.py
- EvaluationContext
- coordinate_axis
- BackendCase
- Implementation
- TestAbsoluteUnit
- _native
- _DomainValue
- AnnualCurrencyUnit
- TestContainer
- bilt
- examples

## God Nodes (most connected - your core abstractions)
1. `Tree` - 91 edges
2. `Unit` - 87 edges
3. `Quantity` - 77 edges
4. `Expr` - 76 edges
5. `UnitComposition` - 64 edges
6. `UnitOrder` - 45 edges
7. `Dimension` - 43 edges
8. `Metadata` - 35 edges
9. `Implementation` - 33 edges
10. `symbol_tree()` - 33 edges

## Surprising Connections (you probably didn't know these)
- `Demo` --uses--> `Backend`  [INFERRED]
  tests/core/test_system.py → src/bilt/_backend/_base_backend.py
- `Demo` --uses--> `JAXBackend`  [INFERRED]
  tests/core/test_system.py → src/bilt/_backend/_jax/_backend.py
- `test_registry_binds_domain_values()` --calls--> `DomainRegistry`  [EXTRACTED]
  tests/core/test_domain_registry.py → src/bilt/core/_domain_registry.py
- `test_registry_rejects_inconsistent_dimension_sizes()` --calls--> `DomainRegistry`  [EXTRACTED]
  tests/core/test_domain_registry.py → src/bilt/core/_domain_registry.py
- `_Domains` --uses--> `Interface`  [INFERRED]
  tests/core/test_implementation.py → src/bilt/core/_interface.py

## Import Cycles
- 3-file cycle: `src/bilt/core/_interface.py -> src/bilt/core/_system/_decorator.py -> src/bilt/core/_system/_dispatch.py -> src/bilt/core/_interface.py`
- 5-file cycle: `src/bilt/core/_interface.py -> src/bilt/core/_system/_implementation.py -> src/bilt/core/_structural_analysis/_protocol.py -> src/bilt/core/_structural_analysis/_exhaustive/_backend.py -> src/bilt/core/_realize_analysis/_realize.py -> src/bilt/core/_interface.py`

## Communities (49 total, 7 thin omitted)

### Community 0 - "Domain"
Cohesion: 0.06
Nodes (48): NotImplementedError, Signature, BasicEquation, Equation, _function_name(), args, kwargs, P (+40 more)

### Community 1 - "Tree"
Cohesion: 0.06
Nodes (22): NoReturn, as_expr(), as_tree(), Expr, Convert supported values to a symbolic Tree., Convert supported values to Expr., Thin user-facing symbolic expression wrapper. Expr exists to make Python syntax…, relation_lhs() (+14 more)

### Community 2 - "Quantity"
Cohesion: 0.09
Nodes (4): Self, Quantity, parametrize, TestQuantity

### Community 3 - "_units.py"
Cohesion: 0.09
Nodes (16): operator, Causal and acausal systems models in Python., unique, UnitCategory, DimensionalityError, MetadataError, Exception, UnitError (+8 more)

### Community 4 - "test_atoms.py"
Cohesion: 0.07
Nodes (30): AttributeError, Hashable, atom_cache_size(), AtomType, clear_atom_cache(), Return the number of currently live interned atoms. This is mainly useful for…, Clear the atom cache. This is mainly useful in tests. Normal code should not…, Identity-based metadata tag that distinguishes kinds of atom. Two AtomType… (+22 more)

### Community 5 - "Unit"
Cohesion: 0.10
Nodes (4): Unit, parametrize, TestDimensionless, TestUnit

### Community 6 - ".symbol"
Cohesion: 0.12
Nodes (31): isolate(), Return a relation tree with ``target`` isolated on the left-hand side.…, int_tree(), test_add_builds_add_tree(), test_as_expr_converts_supported_values(), test_as_tree_converts_supported_values(), test_as_tree_rejects_bool(), test_eq_builds_relation_tree() (+23 more)

### Community 7 - "symbol_tree"
Cohesion: 0.13
Nodes (33): symbol(), symbol_tree(), contains(), Return True if target occurs anywhere inside root. Uses hash-consing by…, parametrize, test_compound_tree_argument_order_matters(), test_compound_tree_head_matters(), test_compound_tree_is_hash_consed() (+25 more)

### Community 8 - "Dimension"
Cohesion: 0.09
Nodes (10): Dimension, Self, parametrize, test_dimension_arithmetic(), test_dimension_caching(), test_dimension_repr(), test_dimension_str(), test_is_derived() (+2 more)

### Community 9 - "test_system.py"
Cohesion: 0.13
Nodes (25): field(), Demo, parametrize, Transform computes an intermediate that is consumed by the equation., Equation computes an intermediate that is consumed by the transform., test_can_access_set_field(), test_can_instantiate_with_a_subset_of_fields(), test_can_instantiate_with_all_fields() (+17 more)

### Community 10 - "UnitOrder"
Cohesion: 0.10
Nodes (3): Self, UnitOrder, TestUnitOrder

### Community 11 - "_isolate.py"
Cohesion: 0.12
Nodes (22): IsolationError, Exception, Base class for symbolic isolation failures., Raised when the target does not occur in the equation., Raised when the target occurs on both sides of the equation., TargetNotFoundError, TargetOnBothSidesError, assert_binary_relation_tree() (+14 more)

### Community 12 - "DomainRegistry"
Cohesion: 0.13
Nodes (17): _DimensionBinding, DomainRegistry, DomainRegistryError, Coordinate, ValueError, Return the position of a coordinate on a dimension., Raised when dimensions have conflicting evaluation bindings., Dimension sizes and coordinates for one evaluation. (+9 more)

### Community 14 - "Atom"
Cohesion: 0.12
Nodes (17): _build_equation_tree(), _build_symbols(), _equation_return_error_message(), Atom, floating(), integer(), Return the built-in relation operator atom for a symbolic operator string., A hash-consed immutable leaf value. Every Atom has an AtomType and a value.… (+9 more)

### Community 15 - "_expr_ops.py"
Cohesion: 0.22
Nodes (25): Raised when the target occurs in a form the isolator cannot handle., UnsupportedIsolationError, expr_args(), expr_contains(), make_add(), make_div(), make_inv(), make_mul() (+17 more)

### Community 16 - "core/__init__.py"
Cohesion: 0.17
Nodes (12): BiltCoreError, EquationReturnError, Exception, Raised when an equation function does not return a Relation., Systems models using transforms., Attribute, _dispatch_evaluation(), T (+4 more)

### Community 17 - "JAXBackend"
Cohesion: 0.15
Nodes (17): NodeTransformer, JAXBackend, FunctionDef, JAX backend implementation of the Bilt backend interface., Compile a single transform AST into a callable function., Bilt JAX backend package., jax_backend(), parse_to_ast() (+9 more)

### Community 18 - "_exhaustive/_backend.py"
Cohesion: 0.22
Nodes (11): BiltModelConstituent, BiltSystemAnalysisError, exhaustive_structural_backend(), Collection, Perform an exhaustive analysis of the model constituents to find a plan to…, get_current_structural_analysis_backend(), Collection, Get the current structural analysis backend. (+3 more)

### Community 19 - "Metadata"
Cohesion: 0.12
Nodes (5): Metadata, Self, _create_name(), parametrize, TestMetadata

### Community 20 - "Interface"
Cohesion: 0.22
Nodes (18): Any, dataclass_transform, Interface, _create_bilt_system_class_attribute(), _create_decorator(), _create_eq_method(), _create_initialize_method(), _create_method() (+10 more)

### Community 21 - "symbolic/_dimensions.py"
Cohesion: 0.17
Nodes (7): DomainLike, CoordinateAxis, Ordered coordinates associated with a symbolic dimension., as_domain(), Tools for symbolic equation analysis., test_as_domain_converts_supported_values(), test_scalar_domain()

### Community 22 - "_to_ast.py"
Cohesion: 0.20
Nodes (14): equation_to_ast(), FunctionDef, _fold_binop(), isolated_tree_to_functiondef(), expr, FunctionDef, _tree_to_ast_expr(), IsolatedTree (+6 more)

### Community 23 - "test_implementation.py"
Cohesion: 0.16
Nodes (9): MonkeyPatch, _Domains, evaluation(), _EvaluationContext, fixture, Self, _System, test_evaluate_binds_computed_output() (+1 more)

### Community 24 - "Tensor"
Cohesion: 0.18
Nodes (6): OverArg, _arithmetic_not_implemented(), _domain_from_over(), Concrete values over a Bilt domain., Create a Bilt tensor from concrete user data. Scalars may omit a domain. Non-…, Tensor

### Community 25 - "Backend"
Cohesion: 0.21
Nodes (10): Backend, Self, Interface for methods to implement by the JAX backend., Must at minimum initialise the ast_transformer and compilation_namespace…, _operation_to_callable(), Convert an AnalysisPlan into a BackendCase by compiling each step to a callable., RealizationFailure, realize_plan() (+2 more)

### Community 26 - "BackendCallable"
Cohesion: 0.23
Nodes (6): Call, Minimal interface for bilt backend, that will be utilised by the analysis layer., BackendCallable, Interface for methods to implement by the JAX backend., JAXASTTransformer, FunctionDef

### Community 27 - "_tensors.py"
Cohesion: 0.16
Nodes (6): ndarray, ValueError, Raised when concrete tensor values do not match their declared domain., TensorDomainError, _check_rank(), test_tensor_rejects_incorrect_rank()

### Community 28 - "PythonBackend"
Cohesion: 0.21
Nodes (12): PythonBackend, Python backend implementation of the Bilt backend interface., parse_to_ast(), fixture, FunctionDef, parametrize, python_backend(), Tests for transform_to_jax module. (+4 more)

### Community 29 - "test_equation.py"
Cohesion: 0.19
Nodes (6): equation, relationship(), test_equation_error_mentions_function_name(), test_equation_error_mentions_return_type(), test_equation_raises_if_function_returns_expr(), test_equation_raises_if_function_returns_non_relation_tree()

### Community 30 - "_tube_geometry.py"
Cohesion: 0.24
Nodes (10): element_length(), tube_flow_area(), tube_inner_diameter(), tube_inner_radius(), tube_outer_area_element(), test_element_length(), test_tube_flow_area(), test_tube_inner_diameter() (+2 more)

### Community 31 - "_supports.py"
Cohesion: 0.23
Nodes (9): FunctionDef, Used to compile a single transform AST into a callable function, that can then…, BiltBackendError, CompilationError, Exception, UnsupportedNodeError, FunctionDef, Checks an AST for unsupported nodes. Raises an exception if any unsupported… (+1 more)

### Community 32 - "EvaluationContext"
Cohesion: 0.23
Nodes (7): current_evaluation(), EvaluationContext, Self, State shared across one Bilt evaluation., Return the currently active evaluation., test_current_evaluation_requires_active_context(), test_nested_contexts_share_outer_evaluation()

### Community 33 - "coordinate_axis"
Cohesion: 0.18
Nodes (8): coordinate_axis(), Coordinate, Associate ordered coordinate values with a dimension., test_registry_binds_coordinate_axis(), test_registry_binds_domain_values(), test_registry_rejects_conflicting_coordinates(), test_registry_rejects_inconsistent_dimension_sizes(), test_coordinate_axis_binds_coordinates_to_dimension()

### Community 34 - "BackendCase"
Cohesion: 0.24
Nodes (8): Execute the compiled function with the provided keyword arguments., BackendCase, Container for information expected by the backend to allow it to chain…, Chain a BackendCase into a single callable function., parametrize, test_build_pipeline_for_cases(), test_can_accept_args_or_kwargs(), test_pipeline_rejects_unexpected_input()

### Community 35 - "Implementation"
Cohesion: 0.20
Nodes (3): Implementation, Self, Register a collection of transforms as an implementation of a system.

### Community 37 - "_native"
Cohesion: 0.33
Nodes (4): Bound, PyModule, PyResult, _native()

## Knowledge Gaps
- **2 isolated node(s):** `examples`, `bilt`
  These have ≤1 connection - possible missing edges or undocumented components.
- **7 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Tree` connect `Tree` to `Domain`, `test_atoms.py`, `.symbol`, `symbol_tree`, `_isolate.py`, `Atom`, `_expr_ops.py`, `_to_ast.py`?**
  _High betweenness centrality (0.208) - this node is a cross-community bridge._
- **Why does `_fold_binop()` connect `_to_ast.py` to `Tree`, `_units.py`, `_expr_ops.py`?**
  _High betweenness centrality (0.153) - this node is a cross-community bridge._
- **Why does `Quantity` connect `Quantity` to `_units.py`, `TestAbsoluteUnit`, `Unit`, `UnitOrder`, `UnitComposition`, `Metadata`?**
  _High betweenness centrality (0.104) - this node is a cross-community bridge._
- **Are the 10 inferred relationships involving `Tree` (e.g. with `BasicEquation` and `Equation`) actually correct?**
  _`Tree` has 10 INFERRED edges - model-reasoned connections that need verification._
- **Are the 13 inferred relationships involving `Unit` (e.g. with `AnnualCurrencyUnit` and `Metadata`) actually correct?**
  _`Unit` has 13 INFERRED edges - model-reasoned connections that need verification._
- **Are the 14 inferred relationships involving `Quantity` (e.g. with `Metadata` and `UnitCategory`) actually correct?**
  _`Quantity` has 14 INFERRED edges - model-reasoned connections that need verification._
- **Are the 9 inferred relationships involving `Expr` (e.g. with `BasicEquation` and `Equation`) actually correct?**
  _`Expr` has 9 INFERRED edges - model-reasoned connections that need verification._
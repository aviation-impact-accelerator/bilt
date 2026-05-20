from __future__ import annotations
from dataclasses import dataclass
from .equations import Equation
from .expressions import Expr
from .solve import CannotIsolate
from .transforms import Transform, TransformGraph

class PlanningError(ValueError): pass
class NoPlanError(PlanningError): pass
class AmbiguousPlanError(PlanningError): pass

@dataclass(frozen=True)
class SolveResult:
    requested_outputs: dict[str, float]
    all_values: dict[str, float]
    graph: TransformGraph
    def __getitem__(self, key: str) -> float: return self.requested_outputs[key]
    def __str__(self) -> str: return str(self.requested_outputs)

@dataclass(frozen=True)
class PlanStep:
    equation: Equation
    output: str
    expression: Expr
    required_inputs: frozenset[str]
    def as_transform(self) -> Transform: return self.equation.as_transform(inputs=tuple(sorted(self.required_inputs)), outputs=(self.output,))

@dataclass
class EquationModel:
    name: str
    equations: list[Equation]
    def __init__(self, name: str) -> None: self.name = name; self.equations = []
    def add(self, equation_definition: Equation) -> None: self.equations.append(equation_definition)
    def residuals(self) -> list[Expr]: return [eq.residual() for eq in self.equations]
    def unit_report(self) -> str:
        if not self.equations: return f"{self.name}: no equations"
        return "\n\n".join([f"Unit analysis for model {self.name!r}"] + [eq.unit_report() for eq in self.equations])
    def print_unit_analysis(self) -> None: print(self.unit_report())
    def model_symbols(self) -> set[str]:
        symbols: set[str] = set()
        for eq in self.equations: symbols.update(eq.symbols())
        return symbols
    def candidate_steps(self, known: set[str]) -> list[PlanStep]:
        candidates: list[PlanStep] = []
        for eq in self.equations:
            for output in sorted(eq.symbols() - known):
                try: expression = eq.solve_expr_for(output)
                except CannotIsolate: continue
                required = frozenset(expression.symbols())
                if required <= known: candidates.append(PlanStep(eq, output, expression, required))
        return candidates
    def _backward_reachable_symbols(self, targets: set[str]) -> set[str]:
        reachable = set(targets); changed = True
        while changed:
            changed = False
            for eq in self.equations:
                symbols = eq.symbols()
                if symbols & reachable and not symbols <= reachable: reachable.update(symbols); changed = True
        return reachable
    def _useful_candidate_steps(self, *, candidates: list[PlanStep], known: set[str], targets: set[str]) -> list[PlanStep]:
        direct = [c for c in candidates if c.output in targets - known]
        if direct: return direct
        relevant = self._backward_reachable_symbols(targets)
        return [c for c in candidates if c.output in relevant]
    def plan(self, *, knowns: list[str] | tuple[str, ...] | set[str], targets: list[str] | tuple[str, ...] | set[str], require_unique: bool = True) -> TransformGraph:
        known = set(knowns); target_set = set(targets)
        unknown_targets = target_set - self.model_symbols() - known
        if unknown_targets: raise NoPlanError(f"Unknown target symbols: {sorted(unknown_targets)}. Model symbols: {sorted(self.model_symbols())}.")
        transforms: list[Transform] = []; produced: set[str] = set()
        while not target_set <= known:
            candidates = [c for c in self._useful_candidate_steps(candidates=self.candidate_steps(known), known=known, targets=target_set) if c.output not in produced]
            if not candidates: raise NoPlanError(self.explain_no_plan(known=known, targets=target_set))
            output_to_candidates: dict[str, list[PlanStep]] = {}
            for c in candidates: output_to_candidates.setdefault(c.output, []).append(c)
            ambiguous = {o: g for o, g in output_to_candidates.items() if len({c.equation.name for c in g}) > 1}
            if require_unique and ambiguous:
                details = "\n".join(f"  - {c.equation.name} -> {c.output} requires {sorted(c.required_inputs)}" for g in ambiguous.values() for c in g)
                raise AmbiguousPlanError("More than one equation can produce the same needed symbol.\n" + details)
            chosen = candidates[0]; transforms.append(chosen.as_transform()); known.add(chosen.output); produced.add(chosen.output)
        return TransformGraph(transforms)
    def solve(self, *, inputs: dict[str, float], outputs: list[str] | tuple[str, ...] | set[str], require_unique: bool = True) -> SolveResult:
        output_list = list(outputs); graph = self.plan(knowns=set(inputs), targets=set(output_list), require_unique=require_unique); all_values = graph.execute(inputs)
        return SolveResult({output: all_values[output] for output in output_list}, all_values, graph)
    def explain_no_plan(self, *, known: set[str], targets: set[str]) -> str:
        lines = ["Could not build a unique explicit DAG plan.", f"Currently known symbols: {sorted(known)}", f"Missing targets: {sorted(targets-known)}", "", "Available equations:"]
        for eq in self.equations:
            symbols = eq.symbols(); lines.append(f"  - {eq.name}: symbols={sorted(symbols)}, known_here={sorted(symbols & known)}, unknown_here={sorted(symbols - known)}")
        return "\n".join(lines)

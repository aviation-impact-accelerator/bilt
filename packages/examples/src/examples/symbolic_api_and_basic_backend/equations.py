from __future__ import annotations
from dataclasses import dataclass
from inspect import Signature
from typing import TYPE_CHECKING
from .expressions import Constant, Expr, Symbol, expression_unit_report
from .relations import Equality, Relation, UnitAnalysis
from .solve import isolate
if TYPE_CHECKING:
    from .transforms import Transform

@dataclass(frozen=True)
class Equation:
    name: str
    relation: Relation
    symbol_table: dict[str, Symbol]
    signature: Signature
    def residual(self) -> Expr: return self.relation.residual()
    def symbols(self) -> set[str]: return self.relation.symbols()
    def check_units(self) -> None: self.relation.check_units()
    def unit_analysis(self) -> UnitAnalysis: return self.relation.unit_analysis(name=self.name)
    def unit_report(self) -> str: return self.unit_analysis().to_text()
    def residual_unit_report(self) -> str: return expression_unit_report(f"{self.name} residual", self.residual())
    def print_unit_analysis(self) -> None: print(self.unit_report())
    def residual_value(self, values: dict[str, float]) -> float: return self.residual().eval(values)
    def solve_expr_for(self, unknown: str) -> Expr:
        if not isinstance(self.relation, Equality): raise ValueError("Can only isolate variables from equality equations.")
        return isolate(self.residual(), Constant(0.0), unknown)
    def solve_for(self, unknown: str, knowns: dict[str, float]) -> float: return self.solve_expr_for(unknown).eval(knowns)
    def as_transform(self, *, inputs: list[str] | tuple[str, ...], outputs: list[str] | tuple[str, ...]) -> 'Transform':
        from .transforms import Transform
        input_set = set(inputs); expressions: dict[str, Expr] = {}
        for output in outputs:
            expr = self.solve_expr_for(output)
            missing = expr.symbols() - input_set
            if missing: raise ValueError(f"Cannot create transform for output {output!r}. Expression requires missing inputs: {sorted(missing)}")
            expressions[output] = expr
        return Transform(f"{self.name}__to__{'_'.join(outputs)}", tuple(inputs), tuple(outputs), expressions, (self.name,))
    def __str__(self) -> str: return f"{self.name}: {self.relation}"

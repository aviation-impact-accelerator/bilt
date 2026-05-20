from __future__ import annotations
from dataclasses import dataclass
from .expressions import Expr, expression_unit_report

@dataclass(frozen=True)
class Transform:
    name: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    expressions: dict[str, Expr]
    source_equations: tuple[str, ...]
    def eval(self, values: dict[str, float]) -> dict[str, float]:
        missing = set(self.inputs) - set(values)
        if missing: raise KeyError(f"Missing transform inputs: {sorted(missing)}")
        return {output: expression.eval(values) for output, expression in self.expressions.items()}
    def unit_report(self) -> str:
        return "\n".join([f"{self.name}: transform unit analysis"] + [expression_unit_report(f"  {o}", e) for o, e in self.expressions.items()])
    def print_unit_analysis(self) -> None: print(self.unit_report())
    def __str__(self) -> str: return f"{self.name}: ({', '.join(self.inputs)}) -> ({', '.join(self.outputs)})"

@dataclass(frozen=True)
class TransformGraph:
    transforms: list[Transform]
    def execute(self, values: dict[str, float]) -> dict[str, float]:
        result = dict(values)
        for transform in self.transforms: result.update(transform.eval(result))
        return result
    def dependencies(self) -> dict[str, set[str]]:
        return {output: set(transform.inputs) for transform in self.transforms for output in transform.outputs}
    def describe(self) -> str:
        if not self.transforms: return "Empty transform graph."
        lines = ["Generated transform graph:"]
        for i, t in enumerate(self.transforms, 1): lines.append(f"{i}. {t.name}\n   inputs:  {list(t.inputs)}\n   outputs: {list(t.outputs)}")
        return "\n".join(lines)
    def to_mermaid(self) -> str:
        lines = ["flowchart LR"]
        for transform in self.transforms:
            for input_name in transform.inputs: lines.append(f"    {input_name} --> {transform.name}")
            for output_name in transform.outputs: lines.append(f"    {transform.name} --> {output_name}")
        return "\n".join(lines)

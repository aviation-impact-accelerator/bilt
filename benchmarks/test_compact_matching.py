"""Benchmark exact compact structural matching of large regular tensor systems."""

import pytest
from pytest_benchmark.fixture import BenchmarkFixture  # noqa: TC002 - runtime fixture resolution

from bilt.core._domain_registry import DomainRegistry
from bilt.core._structural_analysis._compact_incidence import build_compact_incidence
from bilt.core._structural_analysis._compact_matching import match_compact_incidence
from bilt.symbolic._atoms import MUL
from bilt.symbolic._dimensions import coordinate_axis, dim
from bilt.symbolic._domains import domain
from bilt.symbolic._tree import Tree, equality_tree, sum_tree, symbol_tree

SOURCE = dim("source")
TARGET = dim("target")
DENSE_EQUATION = equality_tree(
    sum_tree(Tree(MUL, (symbol_tree("A"), symbol_tree("x"))), SOURCE),
    symbol_tree("rhs"),
)
POINTWISE_EQUATION = equality_tree(symbol_tree("x"), symbol_tree("rhs"))


@pytest.mark.parametrize("size", (100, 1000, 10000))
@pytest.mark.parametrize("pattern", ("pointwise", "complete"))
def test_compact_matching_scaling(benchmark: BenchmarkFixture, size: int, pattern: str) -> None:
    """Matching must avoid expanding one million or more coordinate edges."""
    bindings = DomainRegistry()
    bindings.bind_axis(coordinate_axis(SOURCE, range(size)))
    bindings.bind_axis(coordinate_axis(TARGET, range(size)))
    if pattern == "complete":
        incidence = build_compact_incidence(
            (DENSE_EQUATION,), {"x": domain(SOURCE)}, {"A": domain(TARGET, SOURCE), "rhs": domain(TARGET)}
        )
    else:
        incidence = build_compact_incidence((POINTWISE_EQUATION,), {"x": domain(SOURCE)}, {"rhs": domain(SOURCE)})

    benchmark.group = "compact-structural-matching"
    benchmark.extra_info["pattern"] = pattern
    benchmark.extra_info["unknown_coordinates"] = size
    benchmark.extra_info["potential_dense_edges"] = size * size if pattern == "complete" else size
    result = benchmark.pedantic(
        match_compact_incidence,
        args=(incidence, bindings),
        kwargs={"max_nodes": 1, "max_edges": 1},
        rounds=5,
        iterations=1,
    )
    assert result.pattern == pattern
    assert result.well_determined
    assert result.residual_coordinates == result.unknown_coordinates == size
    assert result.independent_blocks == (size if pattern == "pointwise" else 1)

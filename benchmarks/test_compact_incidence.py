"""Benchmark compact symbolic incidence without expanding dense tensor graphs."""

import pytest
from pytest_benchmark.fixture import BenchmarkFixture  # noqa: TC002 - runtime fixture resolution

from bilt.core._domain_registry import DomainRegistry
from bilt.core._structural_analysis._compact_incidence import build_compact_incidence
from bilt.symbolic._atoms import MUL
from bilt.symbolic._dimensions import coordinate_axis, dim
from bilt.symbolic._domains import domain
from bilt.symbolic._tree import Tree, equality_tree, sum_tree, symbol_tree

SOURCE = dim("source")
TARGET = dim("target")

EXPRESSION = equality_tree(
    sum_tree(Tree(MUL, (symbol_tree("A"), symbol_tree("x"))), SOURCE),
    symbol_tree("rhs"),
)
UNKNOWN_DOMAINS = {"x": domain(SOURCE)}
KNOWN_DOMAINS = {"A": domain(TARGET, SOURCE), "rhs": domain(TARGET)}


@pytest.mark.parametrize("size", (100, 1000, 10000))
def test_compact_incidence_scaling(benchmark: BenchmarkFixture, size: int) -> None:
    """Keep representation cost independent of a dense matrix's element count."""
    benchmark.group = "compact-structural-incidence"
    benchmark.extra_info["panels"] = size
    benchmark.extra_info["potential_dense_edges"] = size * size
    result = benchmark.pedantic(
        build_compact_incidence,
        args=((EXPRESSION,), UNKNOWN_DOMAINS, KNOWN_DOMAINS),
        rounds=5,
        iterations=1,
    )
    assert result.pattern_count == 1
    assert len(result.residuals) == 1
    assert result.residuals[0].domain == domain(TARGET)

    # Check a single row outside timing. This should be O(n), not O(n²).
    registry = DomainRegistry()
    registry.bind_axis(coordinate_axis(SOURCE, range(size)))
    registry.bind_axis(coordinate_axis(TARGET, range(size)))
    dependencies = result.residuals[0].dependencies((0,), registry)
    assert len(dependencies) == size

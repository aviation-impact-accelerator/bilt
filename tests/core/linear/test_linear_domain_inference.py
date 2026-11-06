import collections
import itertools
import typing

import pytest

from bilt.core._structural_analysis._domain_inference import infer_linear_domains
from bilt.core._structural_analysis._linear_block import BiltLinearBlockError
from bilt.core._tensors._expression_lowering import lower_expression
from bilt.errors import BiltTensorDomainError
from bilt.symbolic._dimensions import Dim
from bilt.symbolic._domains import Domain, domain, union
from bilt.symbolic._expressions import Expr, tensor_symbol_expr

if typing.TYPE_CHECKING:
    from bilt.symbolic._tree import Tree

A = Dim("a")
B = Dim("b")


def _reference_layouts(
    rows: tuple[tuple[Tree, Tree], ...], known: dict[str, Domain], hints: dict[str, Domain], canonical: Domain
) -> list[dict[str, Domain]]:
    domains = [
        Domain(tuple(dim for dim, enabled in zip(canonical.dims, flags, strict=True) if enabled))
        for flags in itertools.product((False, True), repeat=len(canonical.dims))
    ]
    valid = []
    for x, y in itertools.product(domains, repeat=2):
        candidate = {"x": x, "y": y}
        if any(set(candidate[name].dims) != set(hint.dims) for name, hint in hints.items()):
            continue
        try:
            row_domains = [
                frozenset(
                    union(
                        *(
                            lower_expression(
                                Expr(tree), symbol_domains=known | candidate, canonical_order=canonical
                            ).domain
                            for tree in row
                        )
                    ).dims
                )
                for row in rows
            ]
        except BiltTensorDomainError:
            continue
        if collections.Counter(row_domains) == collections.Counter(frozenset(d.dims) for d in candidate.values()):
            valid.append(candidate)
    return valid


@pytest.mark.parametrize("canonical", (domain(A, B), domain(B, A)), ids=("ab", "ba"))
def test_inference_agrees_with_exhaustive_domain_lowering(canonical: Domain) -> None:
    x = tensor_symbol_expr("x")
    y = tensor_symbol_expr("y")
    terms = (
        x + y,
        x - y,
        x.sum(A) + y,
        x + y.sum(A),
        (x + y).sum(A),
        x.sum(B) + y,
        (x + y).isel(A, 0),
        x.sum(A).mean(B) + y,
    )
    known = {"a": domain(A), "b": domain(B), "grid": domain(A, B), "scalar": Domain.scalar()}
    known_names = tuple(known)
    choices = ({}, {"x": Domain.scalar()}, {"y": domain(A)}, {"x": domain(A, B)})
    for first, second in itertools.product(range(len(terms)), repeat=2):
        rows = (
            (terms[first].tree, Expr.symbol(known_names[first % 4]).tree),
            (terms[second].tree, Expr.symbol(known_names[second % 4]).tree),
        )
        hints = choices[(first + second) % len(choices)]
        valid = _reference_layouts(rows, known, hints, canonical)
        least = [
            layout
            for layout in valid
            if all(set(layout[name].dims) <= set(other[name].dims) for other in valid for name in ("x", "y"))
        ]
        if least:
            actual = infer_linear_domains(("x", "y"), rows, known, hints, canonical)
            assert {name: set(value.dims) for name, value in actual.items()} == {
                name: set(value.dims) for name, value in least[0].items()
            }
        else:
            message = "Ambiguous" if valid else "No consistent"
            with pytest.raises(BiltLinearBlockError, match=message):
                infer_linear_domains(("x", "y"), rows, known, hints, canonical)


def test_row_and_unknown_order_do_not_choose_the_domain_layout() -> None:
    x = tensor_symbol_expr("x")
    y = tensor_symbol_expr("y")
    rows = ((x.sum(A).tree, Expr.symbol("q").tree), ((x + y).tree, Expr.symbol("p").tree))
    known = {"p": domain(A), "q": Domain.scalar()}
    for names in (("x", "y"), ("y", "x")):
        for order in (rows, rows[::-1]):
            assert infer_linear_domains(names, order, known, {}, domain(A)) == {"x": domain(A), "y": Domain.scalar()}


def test_counting_axes_separately_does_not_prove_a_square_layout() -> None:
    # One A row and one B row cannot cover one A+B unknown and one scalar,
    # even though both sides have one A axis and one B axis in total.
    rows = ((Expr.symbol("a").tree, Expr.integer(0).tree), (Expr.symbol("b").tree, Expr.integer(0).tree))
    with pytest.raises(BiltLinearBlockError, match="No consistent"):
        infer_linear_domains(
            ("x", "y"),
            rows,
            {"a": domain(A), "b": domain(B)},
            {"x": domain(A, B), "y": Domain.scalar()},
            domain(A, B),
        )


def test_many_required_axes_are_propagated_without_enumerating_layouts() -> None:
    names = (*tuple(f"x{index}" for index in range(24)), "y")
    tensors = tuple(tensor_symbol_expr(name) for name in names[:-1])
    y = tensor_symbol_expr("y")
    rows = [(value + y, Expr.symbol("p")) for value in tensors]
    reduced = tensors[0].sum(A).mean(B)
    for value in tensors[1:]:
        reduced = reduced + value.sum(A).mean(B)
    rows.append((reduced - y, Expr.symbol("q")))
    actual = infer_linear_domains(
        names,
        tuple((left.tree, right.tree) for left, right in rows),
        {"p": domain(A, B), "q": Domain.scalar()},
        {},
        domain(A, B),
    )
    assert actual == {**dict.fromkeys(names[:-1], domain(A, B)), "y": Domain.scalar()}


def test_equal_named_dimension_instances_have_the_same_constraints() -> None:
    x = tensor_symbol_expr("x")
    y = tensor_symbol_expr("y")
    rows = (((x + y).tree, Expr.symbol("p").tree), ((x.sum(A) - y).tree, Expr.symbol("q").tree))
    fresh_axis = Dim("a")
    assert fresh_axis == A
    assert fresh_axis is not A
    assert infer_linear_domains(
        ("x", "y"), rows, {"p": domain(fresh_axis), "q": Domain.scalar()}, {}, domain(fresh_axis)
    ) == {"x": domain(fresh_axis), "y": Domain.scalar()}

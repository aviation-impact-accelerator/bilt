"""Differential checks for the experimental native planner."""

from bilt._native import NativeStructuralModel


def test_joint_chain() -> None:
    model = NativeStructuralModel([[(0, [0, 1], None, None), (1, [1, 2], None, None)]])
    assert model.plan([0], [1, 2], []) == [(0, 1, []), (1, 2, [])]
    assert model.plan([], [2], []) is None
    assert model.plan([0], [2], [(0, 1)]) is None


def test_nested_boundary() -> None:
    model = NativeStructuralModel([[(0, [0, 1], None, 1), (1, [1, 2], None, None)], [(0, [0, 1], None, None)]])
    assert model.plan([0], [2], []) == [(0, 1, [0]), (1, 2, [])]


def test_overconstrained_and_forward_only() -> None:
    duplicate = NativeStructuralModel([[(0, [0, 1], None, None), (1, [0, 1], None, None)]])
    assert duplicate.plan([0], [1], []) is None
    transform = NativeStructuralModel([[(0, [0, 1], 1, None)]])
    assert transform.plan([1], [0], []) is None

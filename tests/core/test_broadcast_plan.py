import pytest

from bilt.core._system._analysis import BroadcastPlan, TensorPlan, build_broadcast_plan
from bilt.core._system._domain_registry import DomainShapeError
from bilt.core._system._evaluation import domain_registry_scope
from bilt.core._tensor._tensor import tensor
from bilt.symbolic._dim import Dim
from bilt.symbolic._domains import Domain


def test_two_1d_tensors_on_different_dimensions() -> None:
    time = Dim("time")
    node = Dim("node")

    with domain_registry_scope() as registry:
        a = tensor([1.0, 2.0, 3.0], over=time)
        b = tensor([10.0, 20.0], over=node)

        a.observe(registry)
        b.observe(registry)

        broadcast_plan = build_broadcast_plan({"a": a, "b": b}, registry)

        assert broadcast_plan == BroadcastPlan(
            [
                TensorPlan(
                    name="a",
                    dimension_indices_in_output=(0,),
                ),
                TensorPlan(
                    name="b",
                    dimension_indices_in_output=(1,),
                ),
            ],
            output_domain=Domain(
                (
                    Dim("time"),
                    Dim("node"),
                )
            ),
        )


def test_2d_and_1d_tensor_with_shared_dimension() -> None:
    time = Dim("time")
    node = Dim("node")

    with domain_registry_scope() as registry:
        a = tensor(
            [
                [1.0, 2.0],
                [3.0, 4.0],
                [5.0, 6.0],
            ],
            over=(time, node),
        )

        b = tensor([10.0, 20.0], over=node)

        a.observe(registry)
        b.observe(registry)

        broadcast_plan = build_broadcast_plan({"a": a, "b": b}, registry)

        assert broadcast_plan == BroadcastPlan(
            [
                TensorPlan(
                    name="a",
                    dimension_indices_in_output=(0, 1),
                ),
                TensorPlan(
                    name="b",
                    dimension_indices_in_output=(1,),
                ),
            ],
            output_domain=Domain(
                (
                    Dim("time"),
                    Dim("node"),
                )
            ),
        )


def test_two_2d_tensors_with_overlapping_dimension() -> None:
    time = Dim("time")
    node = Dim("node")
    scenario = Dim("scenario")

    with domain_registry_scope() as registry:
        a = tensor(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
            ],
            over=(time, node),
        )

        b = tensor(
            [
                [10.0, 100.0],
                [20.0, 200.0],
                [30.0, 300.0],
            ],
            over=(node, scenario),
        )

        a.observe(registry)
        b.observe(registry)

        broadcast_plan = build_broadcast_plan({"a": a, "b": b}, registry)

        assert broadcast_plan == BroadcastPlan(
            [
                TensorPlan(
                    name="a",
                    dimension_indices_in_output=(0, 1),
                ),
                TensorPlan(
                    name="b",
                    dimension_indices_in_output=(1, 2),
                ),
            ],
            output_domain=Domain(
                (
                    Dim("time"),
                    Dim("node"),
                    Dim("scenario"),
                )
            ),
        )


def test_two_1d_tensors_with_same_size_but_different_dimensions() -> None:
    time = Dim("time")
    node = Dim("node")

    with domain_registry_scope() as registry:
        a = tensor([1.0, 2.0, 3.0], over=time)
        b = tensor([10.0, 20.0, 30.0], over=node)

        a.observe(registry)
        b.observe(registry)

        broadcast_plan = build_broadcast_plan({"a": a, "b": b}, registry)

        assert broadcast_plan == BroadcastPlan(
            [
                TensorPlan(
                    name="a",
                    dimension_indices_in_output=(0,),
                ),
                TensorPlan(
                    name="b",
                    dimension_indices_in_output=(1,),
                ),
            ],
            output_domain=Domain(
                (
                    Dim("time"),
                    Dim("node"),
                )
            ),
        )


def test_scalar_omitted_from_broadcast_plan() -> None:
    time = Dim("time")
    node = Dim("node")

    with domain_registry_scope() as registry:
        a = tensor([1.0, 2.0, 3.0], over=time)
        b = tensor([10.0, 20.0], over=node)
        scale = 1.0

        a.observe(registry)
        b.observe(registry)

        broadcast_plan = build_broadcast_plan({"a": a, "b": b, "scale": scale}, registry)

        assert broadcast_plan == BroadcastPlan(
            [
                TensorPlan(
                    name="a",
                    dimension_indices_in_output=(0,),
                ),
                TensorPlan(
                    name="b",
                    dimension_indices_in_output=(1,),
                ),
            ],
            output_domain=Domain(
                (
                    Dim("time"),
                    Dim("node"),
                )
            ),
        )


def test_two_2d_tensors_with_same_dimensions_with_different_size_but_one_transposed() -> None:
    time = Dim("time")
    node = Dim("node")

    with domain_registry_scope() as registry:
        a = tensor(
            [
                [1.0, 2.0],
                [3.0, 4.0],
                [5.0, 6.0],
            ],
            over=(time, node),
        )

        b = tensor(
            [
                [10.0, 20.0, 30.0],
                [40.0, 50.0, 60.0],
            ],
            over=(node, time),
        )

        a.observe(registry)
        b.observe(registry)

        broadcast_plan = build_broadcast_plan({"a": a, "b": b}, registry)

        assert broadcast_plan == BroadcastPlan(
            [
                TensorPlan(
                    name="a",
                    dimension_indices_in_output=(0, 1),
                ),
                TensorPlan(
                    name="b",
                    dimension_indices_in_output=(1, 0),
                ),
            ],
            output_domain=Domain(
                (
                    Dim("time"),
                    Dim("node"),
                )
            ),
        )


def test_two_2d_tensors_with_same_dimensions_with_same_size_but_one_transposed() -> None:
    time = Dim("time")
    node = Dim("node")

    with domain_registry_scope() as registry:
        a = tensor(
            [
                [1.0, 2.0],
                [3.0, 4.0],
            ],
            over=(time, node),
        )

        b = tensor(
            [
                [10.0, 20.0],
                [30.0, 40.0],
            ],
            over=(node, time),
        )

        a.observe(registry)
        b.observe(registry)

        broadcast_plan = build_broadcast_plan({"a": a, "b": b}, registry)

        assert broadcast_plan == BroadcastPlan(
            [
                TensorPlan(
                    name="a",
                    dimension_indices_in_output=(0, 1),
                ),
                TensorPlan(
                    name="b",
                    dimension_indices_in_output=(1, 0),
                ),
            ],
            output_domain=Domain(
                (
                    Dim("time"),
                    Dim("node"),
                )
            ),
        )


def test_three_tensors() -> None:
    time = Dim("time")
    node = Dim("node")
    scenario = Dim("scenario")

    with domain_registry_scope() as registry:
        a = tensor(
            [
                [1.0, 2.0],
                [3.0, 4.0],
            ],
            over=(time, node),
        )

        b = tensor(
            [
                [10.0, 20.0],
                [30.0, 40.0],
            ],
            over=(node, scenario),
        )

        c = tensor(
            [
                [100.0, 200.0],
                [300.0, 400.0],
            ],
            over=(scenario, time),
        )

        a.observe(registry)
        b.observe(registry)
        c.observe(registry)

        broadcast_plan = build_broadcast_plan(
            {
                "a": a,
                "b": b,
                "c": c,
            },
            registry,
        )

        assert broadcast_plan == BroadcastPlan(
            [
                TensorPlan(
                    name="a",
                    dimension_indices_in_output=(0, 1),
                ),
                TensorPlan(
                    name="b",
                    dimension_indices_in_output=(1, 2),
                ),
                TensorPlan(
                    name="c",
                    dimension_indices_in_output=(2, 0),
                ),
            ],
            output_domain=Domain(
                (
                    Dim("time"),
                    Dim("node"),
                    Dim("scenario"),
                )
            ),
        )


def test_separately_constructed_dimensions_with_same_name_align() -> None:
    first_time = Dim("time")
    second_time = Dim("time")

    assert first_time == second_time

    with domain_registry_scope() as registry:
        a = tensor([1.0, 2.0, 3.0], over=first_time)
        b = tensor([10.0, 20.0, 30.0], over=second_time)

        a.observe(registry)
        b.observe(registry)

        broadcast_plan = build_broadcast_plan({"a": a, "b": b}, registry)

        assert broadcast_plan == BroadcastPlan(
            [
                TensorPlan(
                    name="a",
                    dimension_indices_in_output=(0,),
                ),
                TensorPlan(
                    name="b",
                    dimension_indices_in_output=(0,),
                ),
            ],
            output_domain=Domain((Dim("time"),)),
        )


def test_conflicting_sizes_for_same_dimension_raise() -> None:
    time = Dim("time")

    with domain_registry_scope() as registry:
        a = tensor([1.0, 2.0, 3.0], over=time)
        b = tensor([10.0, 20.0], over=time)

        a.observe(registry)

        with pytest.raises(DomainShapeError):
            b.observe(registry)


def test_size_one_does_not_broadcast_over_same_named_dimension() -> None:
    time = Dim("time")

    with domain_registry_scope() as registry:
        a = tensor([1.0], over=time)
        b = tensor([10.0, 20.0, 30.0], over=time)

        a.observe(registry)

        with pytest.raises(DomainShapeError):
            b.observe(registry)

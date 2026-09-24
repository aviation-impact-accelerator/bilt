"""Tests for the naming a `BackendCallable` puts on the values its compiled function returns."""

from bilt._backend import BackendCallable


def test_a_callable_with_no_outputs_reports_its_one_value_under_its_name() -> None:
    callable_ = BackendCallable(name="c", parameters=("a", "b"), callable=lambda a, b: a + b)

    assert callable_.evaluate({"a": 1.0, "b": 2.0, "unrelated": 9.0}) == {"c": 3.0}


def test_a_callable_with_outputs_names_the_values_in_the_order_they_are_returned() -> None:
    callable_ = BackendCallable(
        name="e",
        parameters=("a",),
        callable=lambda a: (a + 1.0, a + 2.0),
        outputs=("c", "e"),
    )

    assert callable_.evaluate({"a": 1.0}) == {"c": 2.0, "e": 3.0}


def test_a_single_named_output_is_still_returned_as_a_tuple() -> None:
    callable_ = BackendCallable(name="c", parameters=("a",), callable=lambda a: (a + 1.0,), outputs=("c",))

    assert callable_.evaluate({"a": 1.0}) == {"c": 2.0}

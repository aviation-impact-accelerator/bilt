from bilt.core import transform


def test_transform_decorator_adds_bilt_metadata() -> None:
    @transform
    def c(a: float, b: float) -> float:
        return a + b

    result = 3.0

    assert c.__bilt_name__ == "c"
    assert c.__bilt_parameters__ == ("a", "b")
    assert c(1.0, 2.0) == result

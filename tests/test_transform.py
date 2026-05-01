from bilt._transform import BasicTransform, transform


def test_basic_transform() -> None:
    @transform
    def c(a: float, b: float) -> float:
        return a + b

    assert isinstance(c, BasicTransform)
    assert c.__bilt_transform_name__ == "c"
    assert c.__bilt_transform_parameters__ == ("a", "b")

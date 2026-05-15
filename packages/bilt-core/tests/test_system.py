from bilt.core._system._decorator import system


def test_decorator() -> None:
    @system
    class Demo:
        a: float
        b: float
        c: float

    assert Demo.__name__ == "Demo"

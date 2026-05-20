from bilt.core._system._sentinel import _Missing

class Demo:
    def __init__(
        self,
        a: float | _Missing = ...,
        b: float | _Missing = ...,
        c: float | _Missing = ...,
    ) -> None: ...
    @property
    def a(self) -> float: ...
    @property
    def b(self) -> float: ...
    @property
    def c(self) -> float: ...
    def __eq__(self, other: object) -> bool: ...
    __hash__: None  # type: ignore[assignment]

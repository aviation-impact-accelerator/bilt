import typing

type Array = typing.Any


class Numerics(typing.Protocol):
    def stack(
        self,
        values: typing.Sequence[Array],
        *,
        axis: int,
    ) -> Array: ...

    def solve(
        self,
        matrix: Array,
        rhs: Array,
    ) -> Array: ...

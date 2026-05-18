import dataclasses
import typing

from bilt.units import Quantity, gram, meter


class TestContainer:
    @staticmethod
    def test_heterogeneous_container() -> None:
        @dataclasses.dataclass
        class HeterogeneousContainer:
            mass: typing.Annotated[Quantity, gram]
            length: typing.Annotated[Quantity, meter]

        _ = HeterogeneousContainer(1.0 * gram, 2.0 * meter)

    @staticmethod
    def test_homogeneous_container() -> None:
        @dataclasses.dataclass
        class HomogeneousContainer[T: Quantity]:
            first: T
            second: T

        _ = HomogeneousContainer[typing.Annotated[Quantity, meter]](1.0 * meter, 1.0 * meter)

__all__ = ["BiltSystemEvaluationError", "Implementation"]

import types
import typing

from bilt.core._error import BiltCoreError

if typing.TYPE_CHECKING:
    import collections.abc

    from bilt.core._transform import Transform

    from ._interface import Interface


class BiltSystemEvaluationError(BiltCoreError):
    pass


class Implementation:
    """Register a collection of transforms as an implementation of a system."""

    def __init__(self, system: Interface, transforms: collections.abc.Iterable[Transform[typing.Any, ...]]) -> None:
        self._system = system
        self._transforms = types.MappingProxyType({transform.__bilt_name__: transform for transform in transforms})
        system.__bilt_system_registry__.add(self)

    @property
    def system(self) -> Interface:
        return self._system

    @property
    def transforms(self) -> types.MappingProxyType[str, Transform[typing.Any, ...]]:
        return self._transforms

    def evaluate(self, inputs: dict[str, object], output: str) -> object:
        if output in inputs:
            return inputs[output]

        if output not in self.transforms:
            message = f"No transform registered to produce output {output!r}"
            raise BiltSystemEvaluationError(message)
        transform = self.transforms[output]

        parameters = transform.__bilt_parameters__
        for parameter in parameters:
            if parameter not in inputs:
                inputs[parameter] = self.evaluate(inputs=inputs, output=parameter)

        arguments = {parameter: inputs[parameter] for parameter in parameters}
        return transform(**arguments)

    def __enter__(self) -> typing.Self:
        self._system.__bilt_system_implementation_default__.append(self)
        return self

    def __exit__(self, *_: object) -> None:
        self._system.__bilt_system_implementation_default__.pop()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Implementation):
            return NotImplemented
        return self.system == other.system and self.transforms == other.transforms

    def __hash__(self) -> int:
        return hash((self.system, frozenset(self.transforms.items())))

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(system={self.system}, transforms={frozenset(self.transforms.values())!r})"

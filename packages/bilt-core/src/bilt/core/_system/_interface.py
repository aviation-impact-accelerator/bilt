__all__ = ["Implementation", "Interface"]

import dataclasses
import typing

from ._error import BiltUnknownSystemAttributeError

if typing.TYPE_CHECKING:
    import collections.abc

    from bilt.core._transform import Transform


class _Missing:
    def __repr__(self) -> str:
        return "MISSING"


MISSING = _Missing()


class BiltSystemEvaluationError(Exception):
    pass


class Implementation:
    def __init__(self, system: Interface, transforms: collections.abc.Iterable[Transform[typing.Any, ...]]) -> None:
        self._system = system
        self._transforms = {transform.__bilt_transform_name__: transform for transform in transforms}
        system.__bilt_system_registry__.add(self)

    @property
    def system(self) -> Interface:
        return self._system

    @property
    def transforms(self) -> dict[str, Transform[typing.Any, ...]]:
        return self._transforms

    def evaluate(self, inputs: dict[str, object], output: object) -> object:
        if output in inputs:
            return inputs[output]

        if output not in self.transforms:
            message = f"No transform registered to produce output {output!r}"
            raise BiltSystemEvaluationError(message)
        transform = self.transforms[output]

        parameters = transform.__bilt_transform_parameters__
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


class Interface(type):
    __bilt_system_registry__: set[Implementation]
    __bilt_system_implementation_default__: list[Implementation]

    def __new__(mcls: type[Interface], name: str, bases: tuple[type, ...], namespace: dict[str, object]) -> Interface:
        cls = super().__new__(mcls, name, bases, namespace)
        cls.__bilt_system_registry__ = set()
        cls.__bilt_system_implementation_default__ = []
        return cls


class SystemInstance(typing.Protocol):
    __bilt_system_implementation__: Implementation | _Missing
    __bilt_system_instance_variables__: dict[str, object]


@dataclasses.dataclass(frozen=True, slots=True)
class Attribute[T]:
    name: str
    annotation: type[T]


def _dispatch_evaluation[T](instance: SystemInstance, attribute: Attribute[T]) -> T:
    current = typing.cast("T | _Missing", instance.__bilt_system_instance_variables__.get(attribute.name, MISSING))
    if not isinstance(current, _Missing):
        return current
    implementation = instance.__bilt_system_implementation__
    if isinstance(implementation, _Missing):
        instance_class = typing.cast("Interface", instance.__class__)
        if len(instance_class.__bilt_system_implementation_default__) == 0:
            raise BiltUnknownSystemAttributeError(attribute.name, instance_class)
        implementation = instance_class.__bilt_system_implementation_default__[-1]
        instance.__bilt_system_implementation__ = implementation
    inputs = {
        name: value for name, value in instance.__bilt_system_instance_variables__.items() if value is not MISSING
    }
    output = attribute.name
    result = typing.cast("T", implementation.evaluate(inputs=inputs, output=output))
    instance.__bilt_system_instance_variables__[attribute.name] = result
    return result

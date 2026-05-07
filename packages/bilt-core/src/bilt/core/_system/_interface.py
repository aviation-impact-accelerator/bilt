__all__ = ["Interface"]

import contextlib
import typing

if typing.TYPE_CHECKING:
    import collections.abc

    from bilt.core._transform import Transform


class BiltSystemEvaluationError(Exception):
    pass


class Implementation:
    def __init__(self, system: Interface, transforms: frozenset[Transform[typing.Any, ...]]) -> None:
        self._system = system
        self._transforms = {transform.__bilt_transform_name__: transform for transform in transforms}

    @property
    def system(self) -> Interface:
        return self._system

    @property
    def transforms(self) -> dict[str, Transform[typing.Any, ...]]:
        return self._transforms

    @contextlib.contextmanager
    def as_default(self) -> collections.abc.Iterator[typing.Self]:
        current_implementation_default = self._system.__bilt_system_implementation_default__
        self._system.__bilt_system_implementation_default__ = self
        yield self
        self._system.__bilt_system_implementation_default__ = current_implementation_default

    def evaluate(self, inputs: dict[str, object], output: str) -> object:
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
    __bilt_system_implementation_default__: Implementation | None

    def __new__(mcls: type[Interface], name: str, bases: tuple[type, ...], namespace: dict[str, object]) -> Interface:
        cls = super().__new__(mcls, name, bases, namespace)
        cls.__bilt_system_registry__ = set()
        cls.__bilt_system_implementation_default__ = None
        return cls

    def implementation(cls, *value: Transform[typing.Any, ...]) -> Implementation:
        implementation = Implementation(cls, frozenset(value))
        cls.__bilt_system_registry__.add(implementation)
        return implementation

__all__ = ["Interface"]

import typing

if typing.TYPE_CHECKING:
    from bilt.core._transform import Transform


class BiltSystemEvaluationError(Exception):
    pass


class Implementation:
    def __init__(self, system: Interface, transforms: frozenset[Transform[typing.Any, ...]]) -> None:
        self._system = system
        self._transforms = {transform.__bilt_transform_name__: transform for transform in transforms}
        self._previous_default: Implementation | None = None

    @property
    def system(self) -> Interface:
        return self._system

    @property
    def transforms(self) -> dict[str, Transform[typing.Any, ...]]:
        return self._transforms

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

    def implementation(cls, *value: Transform[typing.Any, ...]) -> Implementation:
        implementation = Implementation(cls, frozenset(value))
        cls.__bilt_system_registry__.add(implementation)
        return implementation

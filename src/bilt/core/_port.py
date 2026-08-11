"""A sub-system port's dotted spelling, e.g. ``resistor.v``."""

__all__ = [
    "DOTTED_SEPARATOR",
    "QualifiedPort",
]

import dataclasses
import typing

if typing.TYPE_CHECKING:
    import collections.abc

DOTTED_SEPARATOR: typing.Final = "."


@dataclasses.dataclass(frozen=True, slots=True)
class QualifiedPort:
    """A port of a sub-system, named as the parent system sees it."""

    sub_system: str
    port: str

    @property
    def dotted(self) -> str:
        """The canonical spelling, e.g. ``resistor.v``."""
        return f"{self.sub_system}{DOTTED_SEPARATOR}{self.port}"

    def __str__(self) -> str:
        return self.dotted

    @staticmethod
    def unqualify(qualified: str, sub_system: str, ports: collections.abc.Container[str]) -> str | None:
        """Return the child-local port name, or None when ``qualified`` is not one of ``sub_system``'s ports."""
        prefix = f"{sub_system}{DOTTED_SEPARATOR}"
        if not qualified.startswith(prefix):
            return None

        port = qualified[len(prefix) :]
        return port if port in ports else None

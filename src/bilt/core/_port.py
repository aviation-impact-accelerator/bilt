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
    """A port of a sub-system, named as the parent system sees it.

    The parent's graph is a flat namespace of attribute names, so a child's ports enter it
    under a qualified spelling. This is the one place that knows what that spelling is.
    """

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
        """Return the child-local port name, or None when ``qualified`` is not one of ``sub_system``'s ports.

        ``ports`` is consulted rather than trusted from the prefix alone so that a name such as
        ``resistor.typo`` is rejected instead of being taken for a port the child does not have.
        """
        prefix = f"{sub_system}{DOTTED_SEPARATOR}"
        if not qualified.startswith(prefix):
            return None

        port = qualified[len(prefix) :]
        return port if port in ports else None

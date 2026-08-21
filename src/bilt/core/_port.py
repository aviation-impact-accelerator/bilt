__all__ = [
    "DOTTED_SEPARATOR",
    "MANGLED_SEPARATOR",
    "QualifiedPort",
    "mangle_inputs",
    "mangle_name",
]

import dataclasses
import typing

if typing.TYPE_CHECKING:
    import collections.abc

DOTTED_SEPARATOR: typing.Final = "."
MANGLED_SEPARATOR: typing.Final = "__"


@dataclasses.dataclass(frozen=True, slots=True)
class QualifiedPort:
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


def mangle_inputs(inputs: collections.abc.Mapping[str, object]) -> dict[str, object]:
    """Relabel caller-facing port names to the mangled ones a compiled callable is bound by."""
    return {mangle_name(name): value for name, value in inputs.items()}


def mangle_name(name: str) -> str:
    """Rewrite a dotted port name into the identifier a backend can compile it as."""
    return name.replace(DOTTED_SEPARATOR, MANGLED_SEPARATOR)

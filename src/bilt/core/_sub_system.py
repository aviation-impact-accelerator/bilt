__all__ = ["SubSystemConstituent"]

import typing

from bilt.core._errors import BiltAnalysisError, BiltNameError
from bilt.core._port import MANGLED_SEPARATOR, QualifiedPort

if typing.TYPE_CHECKING:
    import collections.abc

    from bilt._backend._base_backend import Backend
    from bilt.core._port_declaration import PortDeclaration
    from bilt.core._system._implementation import Implementation


class SubSystemConstituent:
    """A child system embedded in a parent implementation, seen by the parent as one relation."""

    def __init__(self, name: str, implementation: Implementation) -> None:
        validate_name(name)
        self.__bilt_name__ = name
        self.implementation = implementation
        self.interface = implementation.system
        self.__bilt_parameters__ = tuple(self.qualify(port) for port in self.interface.__bilt_parameters__)
        self.__bilt_port_declarations__: dict[str, PortDeclaration] = {
            self.qualify(port): declaration for port, declaration in self.interface.__bilt_port_declarations__.items()
        }

    def qualify(self, port: str) -> str:
        """Return the parent-facing name of one of this sub-system's ports."""
        return QualifiedPort(self.__bilt_name__, port).dotted

    def child_port(self, qualified: str) -> str:
        """Return the child-local port name, raising when this sub-system has no such port."""
        port = QualifiedPort.unqualify(
            qualified,
            self.__bilt_name__,
            self.interface.__bilt_parameters__,
        )
        if port is None:
            msg = f"{self!r} does not have port {qualified!r}."
            raise BiltAnalysisError(msg)

        return port

    def required_inputs_for(
        self,
        qualified_output: str,
        backend: Backend,
        *,
        within: collections.abc.Iterable[str] | None = None,
    ) -> frozenset[str] | None:
        """Return the child-local inputs required to produce ``qualified_output``.

        ``within`` restricts the child to those ports, named as the parent sees them; omitted, it
        may use any of its ports. Either way the output itself is not available as an input.
        """
        from bilt.core._evaluation_options import _using_options  # noqa: PLC0415 - equation import cycle

        child_output = self.child_port(qualified_output)

        reachable = (
            set(self.interface.__bilt_parameters__) if within is None else {self.child_port(port) for port in within}
        )
        available = reachable - {child_output}

        with _using_options(self.implementation.options):
            return self.implementation.analyser_for(backend).required_inputs_for_target(
                available,
                child_output,
            )

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.__bilt_name__!r}, interface={self.interface.__bilt_name__!r})"


def validate_name(name: str) -> None:
    """Reject a sub-system name that would make a flattened port name ambiguous."""
    if MANGLED_SEPARATOR in name:
        msg = f"Sub-system name {name!r} may not contain {MANGLED_SEPARATOR!r}, which is reserved for bilt internals."
        raise BiltNameError(msg)

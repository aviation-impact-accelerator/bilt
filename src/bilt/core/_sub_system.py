__all__ = ["SubSystemConstituent"]

import typing

from bilt._port import QualifiedPort

if typing.TYPE_CHECKING:
    from bilt.core._interface import Interface
    from bilt.core._system._implementation import Implementation


class SubSystemConstituent:
    """A reference to a sub-system within a parent system implementation."""

    def __init__(self, __bilt_name__: str, interface: Interface, implementation: Implementation) -> None:
        self.__bilt_name__ = __bilt_name__
        self.interface = interface
        self.implementation = implementation
        self.__bilt_parameters__ = tuple(self.qualify(port) for port in interface.__bilt_parameters__)

    def port(self, port: str) -> QualifiedPort:
        """Return one of this sub-system's ports, named as the parent system sees it."""
        return QualifiedPort(self.__bilt_name__, port)

    def qualify(self, port: str) -> str:
        """Return the parent-facing name of one of this sub-system's ports."""
        return self.port(port).dotted

    def unqualify(self, qualified: str) -> str | None:
        """Return the child-local port name, or None when it is not one of this sub-system's ports."""
        return QualifiedPort.unqualify(qualified, self.__bilt_name__, self.interface.__bilt_parameters__)

__all__ = ["SubSystemConstituent"]

import typing

from bilt.core._port import QualifiedPort
from bilt.core._structural_analysis._errors import BiltSystemAnalysisError

if typing.TYPE_CHECKING:
    from bilt.core._interface import Interface
    from bilt.core._system._implementation import Implementation


class SubSystemConstituent:
    """A child system embedded in a parent implementation, seen by the parent as one relation."""

    def __init__(self, __bilt_name__: str, interface: Interface, implementation: Implementation) -> None:
        self.__bilt_name__ = __bilt_name__
        self.interface = interface
        self.implementation = implementation
        self.__bilt_parameters__ = tuple(self.qualify(port) for port in interface.__bilt_parameters__)

    def qualify(self, port: str) -> str:
        """Return the parent-facing name of one of this sub-system's ports."""
        return QualifiedPort(self.__bilt_name__, port).dotted

    def unqualify(self, qualified: str) -> str | None:
        """Return the child-local port name, or None when it is not one of this sub-system's ports."""
        return QualifiedPort.unqualify(qualified, self.__bilt_name__, self.interface.__bilt_parameters__)

    def child_port(self, qualified: str) -> str:
        """Return the child-local port name, raising when this sub-system has no such port."""
        port = self.unqualify(qualified)
        if port is None:
            msg = f"{self!r} does not have port {qualified!r}."
            raise BiltSystemAnalysisError(msg)
        return port

    def child_route(self, qualified_output: str) -> tuple[str, frozenset[str]] | None:
        """Return the child-local output name and the child-local ports needed to produce it."""
        child_output = self.child_port(qualified_output)
        available = set(self.interface.__bilt_parameters__) - {child_output}
        required = self.implementation.analyser.required_inputs_for_target(available, child_output)
        if required is None:
            return None
        return child_output, required

    def required_parent_inputs(self, qualified_output: str) -> frozenset[str] | None:
        """Return the ports needed to produce ``qualified_output``, named as the parent sees them."""
        route = self.child_route(qualified_output)
        if route is None:
            return None
        _, required = route
        return frozenset(self.qualify(port) for port in required)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.__bilt_name__!r}, interface={self.interface.__name__!r})"

__all__ = ["SubSystemConstituent"]

import typing

from bilt.core._error import BiltNameError, BiltSystemAnalysisError
from bilt.core._evaluation_request import EvaluationRequest
from bilt.core._port import MANGLED_SEPARATOR, QualifiedPort

if typing.TYPE_CHECKING:
    import collections.abc

    from bilt._backend._base_backend import Backend
    from bilt.core._system._implementation import Implementation


class SubSystemConstituent:
    """A child system embedded in a parent implementation, seen by the parent as one relation."""

    def __init__(self, name: str, implementation: Implementation) -> None:
        validate_name(name)
        self.__bilt_name__ = name
        self.implementation = implementation
        self.interface = implementation.system
        self.__bilt_parameters__ = tuple(self.qualify(port) for port in self.interface.__bilt_parameters__)

    def qualify(self, port: str) -> str:
        """Return the parent-facing name of one of this sub-system's ports."""
        return QualifiedPort(self.__bilt_name__, port).dotted

    def child_port(self, qualified: str) -> str:
        """Return the child-local port name, raising when this sub-system has no such port."""
        port = QualifiedPort.unqualify(qualified, self.__bilt_name__, self.interface.__bilt_parameters__)
        if port is None:
            msg = f"{self!r} does not have port {qualified!r}."
            raise BiltSystemAnalysisError(msg)
        return port

    def request_for(
        self,
        qualified_output: str,
        backend: Backend,
        *,
        within: collections.abc.Iterable[str] | None = None,
    ) -> EvaluationRequest | None:
        """Return the child-local request producing ``qualified_output``, or None if no route exists.

        ``within`` restricts the child to those ports, named as the parent sees them; omitted, it
        may use any of its ports. Either way the port asked for is never one the child may use.
        The request names only the ports it will use.
        """
        child_output = self.child_port(qualified_output)
        reachable = (
            set(self.interface.__bilt_parameters__) if within is None else {self.child_port(port) for port in within}
        )
        available = reachable - {child_output}
        required = self.implementation.analyser_for(backend).required_inputs_for_target(available, child_output)
        if required is None:
            return None
        return EvaluationRequest(output=child_output, available=required)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.__bilt_name__!r}, interface={self.interface.__bilt_name__!r})"


def validate_name(name: str) -> None:
    """Reject a sub-system name that would make a flattened port name ambiguous."""
    if MANGLED_SEPARATOR in name:
        msg = f"Sub-system name {name!r} may not contain {MANGLED_SEPARATOR!r}, which is reserved for bilt internals."
        raise BiltNameError(msg)

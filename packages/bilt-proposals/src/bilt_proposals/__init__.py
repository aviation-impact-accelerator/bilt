"""Importable templates of proposed Bilt APIs, without implementations."""

from bilt_proposals._connectors import Connection, field
from bilt_proposals._errors import ProposalNotImplementedError
from bilt_proposals._inequalities import equation
from bilt_proposals._scopes import Scope
from bilt_proposals._unknowns import ContinuousUnknown

__all__ = [
    "Connection",
    "ContinuousUnknown",
    "ProposalNotImplementedError",
    "Scope",
    "equation",
    "field",
]

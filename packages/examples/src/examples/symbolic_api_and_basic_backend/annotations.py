from __future__ import annotations
from dataclasses import dataclass
@dataclass(frozen=True)
class Quantity:
    """Marker type for Annotated[Quantity, unit]."""
@dataclass(frozen=True)
class SymbolMetadata:
    display_name: str | None = None
    description: str | None = None
def Display(name: str) -> SymbolMetadata: return SymbolMetadata(display_name=name)

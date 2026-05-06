from __future__ import annotations

import enum


@enum.unique
class UnitCategory(enum.Enum):
    BASE = enum.auto()
    NAMED = enum.auto()
    DERIVED = enum.auto()
    ABSOLUTE = enum.auto()

__all__ = ["MISSING"]


class _Missing:
    def __repr__(self) -> str:
        return "MISSING"


MISSING = _Missing()

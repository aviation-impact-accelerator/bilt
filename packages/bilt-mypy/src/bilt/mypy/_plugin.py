"""mypy plugin to make mypy compatible with Bilt."""

__all__ = ["plugin"]

import mypy.plugin


def plugin(version: str) -> type[mypy.plugin.Plugin]:  # noqa: ARG001
    """Entry point provided to mypy for accessing `BiltMyPyPlugin`."""
    return BiltMyPyPlugin


class BiltMyPyPlugin(mypy.plugin.Plugin):
    """mypy plugin to make mypy compatible with Bilt."""

import mypy.plugin

from bilt.mypy import plugin


def test_plugin_returns_mypy_plugin_subclass() -> None:
    result = plugin("1.0.0")
    assert issubclass(result, mypy.plugin.Plugin)

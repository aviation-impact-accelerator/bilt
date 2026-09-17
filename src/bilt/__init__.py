"""Causal and acausal systems models in Python."""

import importlib
import typing

from bilt._backend._python import PYTHON_BACKEND
from bilt.core._equation import equation
from bilt.core._interface import Interface
from bilt.core._system import Implementation, field, system
from bilt.core._tensors._tensors import Tensor, tensor
from bilt.core._transform import transform
from bilt.symbolic._dimensions import CoordinateAxis, Dim, coordinate_axis, dim
from bilt.symbolic._domains import Domain, as_domain, domain

if typing.TYPE_CHECKING:
    from bilt._backend._jax import JAX_BACKEND

__all__ = [
    "JAX_BACKEND",
    "PYTHON_BACKEND",
    "CoordinateAxis",
    "Dim",
    "Domain",
    "Implementation",
    "Interface",
    "Tensor",
    "as_domain",
    "coordinate_axis",
    "dim",
    "domain",
    "equation",
    "field",
    "system",
    "tensor",
    "transform",
]

_OPTIONAL: dict[str, tuple[str, str]] = {
    "JAX_BACKEND": ("bilt._backend._jax", "jax"),
}


def __getattr__(name: str) -> typing.Any:  # noqa: ANN401
    """Import a public name that needs an optional extra on first access.

    Raises:
        AttributeError: If ``name`` is not a public name of this module.
        ImportError: If a third-party dependency of the extra is not installed.
    """
    try:
        module_name, extra = _OPTIONAL[name]
    except KeyError:
        msg = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(msg) from None
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as error:
        if error.name is None or error.name == "bilt" or error.name.startswith("bilt."):
            raise
        msg = f"{name} requires the {extra!r} extra; install it with `pip install 'bilt[{extra}]'`."
        raise ImportError(msg) from error
    return getattr(module, name)

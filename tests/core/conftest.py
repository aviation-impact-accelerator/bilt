"""Common backend fixture for core execution and solver tests."""

import typing

import pytest

from bilt._backend._jax._backend import JAX_BACKEND
from bilt._backend._python._backend import PYTHON_BACKEND

if typing.TYPE_CHECKING:
    from bilt._backend import Backend


@pytest.fixture(params=(PYTHON_BACKEND, JAX_BACKEND), ids=("python", "jax"))
def backend(request: pytest.FixtureRequest) -> Backend:
    """Check backend-sensitive core behaviour against both implementations."""
    return request.param

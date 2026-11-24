"""Compare supported native and Python plans without permitting fallback."""

import typing

import pytest

from bilt import field, system, transform
from bilt.core._interface import Interface
from bilt._backend._jax._backend import JAX_BACKEND
from bilt._backend._python._backend import PYTHON_BACKEND
from bilt.core._structural_analysis._causalization_block_solver import _native_prototype as native
from bilt.core._system._dispatch import _prefetch_outputs
from bilt.core._system._implementation import Implementation

if typing.TYPE_CHECKING:
    from bilt._backend import Backend
    from bilt.core._system._instance import SystemInstance


@pytest.mark.parametrize("backend", [PYTHON_BACKEND, JAX_BACKEND])
def test_single_and_joint_native_parity(backend: Backend, monkeypatch: pytest.MonkeyPatch) -> None:
    @system
    class Demo(metaclass=Interface):
        a: float = field()
        base: float = field()
        left: float = field()
        right: float = field()

    @transform
    def base(a: float) -> float:
        return a + 1

    @transform
    def left(base: float) -> float:
        return base * 2

    @transform
    def right(base: float) -> float:
        return base * 3

    results = []
    before = native.STATS["native_success"]
    for mode in ("python", "rust"):
        monkeypatch.setattr(native, "MODE", mode)
        with backend, Implementation(Demo, [base, left, right]):
            single = Demo(a=4.0)
            joint = Demo(a=4.0)
            assert single.left == pytest.approx(10.0)
            _prefetch_outputs(typing.cast("SystemInstance", joint), ("left", "right"))
            results.append((single.left, joint.left, joint.right, joint.base))
    assert results[0] == pytest.approx(results[1])
    assert native.STATS["native_success"] - before == 2

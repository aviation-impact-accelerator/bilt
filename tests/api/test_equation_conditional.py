# ruff: noqa: SIM108
import pytest

from bilt import Implementation, Interface, equation, field, system
from bilt.errors import BiltDefinitionError


@system
class Shifted(metaclass=Interface):
    x: float = field()
    y: float = field()


SHIFT = 2.0


def test_normal_equation_accepts_statement_level_conditional() -> None:
    @equation
    def relation(x: float, y: float) -> bool:
        if x > 0:
            value = x**2
        else:
            value = -x
        return y == value

    with Implementation(Shifted, [relation]):
        assert Shifted(x=2.0).y == pytest.approx(4.0)
        assert Shifted(x=-2.0).y == pytest.approx(2.0)
    assert relation(2.0, 4.0)
    assert relation(-2.0, 2.0)


def test_conditional_equation_uses_local_bindings_and_module_globals() -> None:
    @equation
    def relation(x: float, y: float) -> bool:
        offset: float = x + SHIFT
        if offset > 0:
            value: float = offset**2
        else:
            value = -offset
        return y == value

    with Implementation(Shifted, [relation]):
        assert Shifted(x=1.0).y == pytest.approx(9.0)
        assert Shifted(x=-3.0).y == pytest.approx(1.0)
    assert "_bilt_conditional" not in globals()


def test_conditional_with_closure_capture_is_rejected() -> None:
    offset = 3.0
    with pytest.raises(BiltDefinitionError, match="captures closure variables"):

        @equation
        def relation(x: float, y: float) -> bool:
            if x > 0:
                value = x + offset
            else:
                value = -x
            return y == value


def test_conditional_with_multiple_returns_is_rejected() -> None:
    with pytest.raises(BiltDefinitionError, match="exactly one return"):

        @equation
        def relation(x: float, y: float) -> bool:
            if x > 0:
                return y == x**2
            return y == -x


def test_sequential_conditionals_are_not_supported() -> None:
    with pytest.raises(BiltDefinitionError, match="at most one if/else"):

        @equation
        def relation(x: float, y: float) -> bool:
            if x > 0:
                first = x**2
            else:
                first = -x
            if x > 1:
                second = first + 1
            else:
                second = first - 1
            return y == second


def test_equation_without_inspectable_source_is_rejected() -> None:
    namespace: dict[str, object] = {}
    exec("def generated(x, y):\n    return y == x + 1\n", namespace)  # noqa: S102
    generated = namespace["generated"]
    assert callable(generated)

    with pytest.raises(BiltDefinitionError, match="Cannot inspect source"):
        equation(generated)


def test_procedural_statements_are_rejected_without_executing_them() -> None:
    calls: list[str] = []
    with pytest.raises(BiltDefinitionError, match="Expr is not supported"):

        @equation
        def relation(x: float, y: float) -> bool:
            calls.append("traced")
            return y == x + 1

    assert calls == []


def test_static_python_branch_is_rejected() -> None:
    enabled = True
    with pytest.raises(BiltDefinitionError, match="symbolic comparison"):

        @equation
        def relation(x: float, y: float) -> bool:
            if enabled:
                value = x + 1
            else:
                value = x + 2
            return y == value


def test_decorator_parentheses_remain_supported() -> None:
    @equation()
    def relation(x: float, y: float) -> bool:
        return y == x + 1

    with Implementation(Shifted, [relation]):
        assert Shifted(x=3.0).y == pytest.approx(4.0)

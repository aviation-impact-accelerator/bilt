import pytest

from bilt._backend._python._backend import PythonBackend
from bilt.core import BiltSystemAnalysisError, Implementation, Interface, field, system
from bilt.core._equation import equation
from bilt.core._evaluation_request import EvaluationRequest
from bilt.core._sub_system import SubSystemConstituent

CURRENT = 2.0
RESISTANCE = 3.0
VOLTAGE = 6.0


def _resistor_implementation() -> Implementation:
    @system
    class Resistor(metaclass=Interface):
        v: float = field()
        i: float = field()
        r: float = field()

    @equation
    def ohms_law(v: float, i: float, r: float) -> bool:
        return v == i * r

    return Implementation(Resistor, [ohms_law])


@pytest.fixture
def sub_system() -> SubSystemConstituent:
    """A resistor whose one equation solves for any of its three ports."""
    return SubSystemConstituent(name="resistor", implementation=_resistor_implementation())


@pytest.fixture
def sealed_sub_system() -> SubSystemConstituent:
    """A sub-system with real ports but no equations, so no port is reachable from any other."""

    @system
    class Sealed(metaclass=Interface):
        a: float = field()
        b: float = field()

    return SubSystemConstituent(name="sealed", implementation=Implementation(Sealed, []))


@pytest.fixture
def board(sub_system: SubSystemConstituent) -> Implementation:
    """A parent whose only component is ``sub_system``.

    The board's own ports stay unconnected: wiring a parent port to a child port needs an
    equation that can take an Interface-typed parameter, which this layer does not provide.
    """

    @system
    class Board(metaclass=Interface):
        supply_voltage: float = field()

    return Implementation(Board, [sub_system])


def test_parameters_are_the_qualified_child_ports(sub_system: SubSystemConstituent) -> None:
    assert sub_system.__bilt_parameters__ == ("resistor.v", "resistor.i", "resistor.r")


def test_the_constituent_keeps_its_reference_name() -> None:
    implementation = _resistor_implementation()

    renamed = SubSystemConstituent(name="r1", implementation=implementation)

    assert renamed.__bilt_name__ == "r1"
    assert renamed.__bilt_parameters__ == ("r1.v", "r1.i", "r1.r")


def test_qualify_is_the_dotted_spelling_of_a_child_port(sub_system: SubSystemConstituent) -> None:
    assert sub_system.qualify("v") == "resistor.v"


def test_child_port_recovers_the_child_local_name(sub_system: SubSystemConstituent) -> None:
    assert sub_system.child_port("resistor.v") == "v"


@pytest.mark.parametrize("qualified", ("v", "capacitor.v", "resistor.typo"))
def test_child_port_rejects_a_name_that_is_not_one_of_its_ports(
    sub_system: SubSystemConstituent, qualified: str
) -> None:
    with pytest.raises(BiltSystemAnalysisError, match="does not have port"):
        sub_system.child_port(qualified)


def test_child_route_names_the_output_and_inputs_in_child_local_terms(sub_system: SubSystemConstituent) -> None:
    assert sub_system.child_route("resistor.v") == EvaluationRequest(output="v", available=frozenset({"i", "r"}))


def test_child_route_is_none_without_an_internal_route(sealed_sub_system: SubSystemConstituent) -> None:
    assert sealed_sub_system.child_route("sealed.a") is None


def test_required_parent_inputs_qualifies_the_child_ports(sub_system: SubSystemConstituent) -> None:
    assert sub_system.required_parent_inputs("resistor.v") == frozenset({"resistor.i", "resistor.r"})


def test_required_parent_inputs_is_none_without_an_internal_route(
    sealed_sub_system: SubSystemConstituent,
) -> None:
    assert sealed_sub_system.required_parent_inputs("sealed.a") is None


def test_repr_names_the_reference_and_its_interface(sub_system: SubSystemConstituent) -> None:
    assert repr(sub_system) == "SubSystemConstituent(name='resistor', interface='Resistor')"


def test_a_parent_solves_a_child_port_through_the_sub_system(board: Implementation) -> None:
    result = board.evaluate({"resistor.i": CURRENT, "resistor.r": RESISTANCE}, "resistor.v", PythonBackend())

    assert result == VOLTAGE


def test_a_parent_solves_a_child_port_in_the_inward_direction(board: Implementation) -> None:
    result = board.evaluate({"resistor.v": VOLTAGE, "resistor.r": RESISTANCE}, "resistor.i", PythonBackend())

    assert result == CURRENT


def test_the_realized_callable_speaks_dotted_names(board: Implementation) -> None:
    request = EvaluationRequest(output="resistor.v", available=frozenset({"resistor.i", "resistor.r"}))

    realized = board.get_evaluation(request, PythonBackend())

    assert realized.parameters == ("resistor.i", "resistor.r")
    assert realized.callable(**{"resistor.i": CURRENT, "resistor.r": RESISTANCE}) == VOLTAGE


def test_nothing_a_backend_compiles_carries_a_dotted_name(board: Implementation) -> None:
    case = board.analyser.analyse({"resistor.i", "resistor.r"}, "resistor.v", PythonBackend())

    assert "." not in case.name
    assert "." not in case.return_name
    assert not any("." in name for name in case.parameters)
    assert not any("." in name for intermediate in case.intermediates for name in intermediate.parameters)


def test_a_parent_reuses_one_cache_entry_across_repeated_evaluations(board: Implementation) -> None:
    backend = PythonBackend()

    board.evaluate({"resistor.i": CURRENT, "resistor.r": RESISTANCE}, "resistor.v", backend)
    board.evaluate({"resistor.i": CURRENT + 1, "resistor.r": RESISTANCE + 1}, "resistor.v", backend)

    assert len(board._evaluation_cache) == 1  # noqa: SLF001


def test_a_parent_cannot_solve_a_child_port_the_child_has_no_route_to(
    sealed_sub_system: SubSystemConstituent,
) -> None:
    @system
    class Rack(metaclass=Interface):
        supply_voltage: float = field()

    rack = Implementation(Rack, [sealed_sub_system])

    with pytest.raises(BiltSystemAnalysisError, match="Cannot compute"):
        rack.evaluate({"sealed.b": 1.0}, "sealed.a", PythonBackend())

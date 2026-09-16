import pytest

from bilt import BiltNameError, BiltSystemAnalysisError, Implementation, Interface, field, system
from bilt._backend._python._backend import PYTHON_BACKEND
from bilt.core._equation import equation
from bilt.core._evaluation_request import EvaluationRequest
from bilt.core._port import DOTTED_SEPARATOR, MANGLED_SEPARATOR, QualifiedPort
from bilt.core._sub_system import SubSystemConstituent, validate_name

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
def spare_port_sub_system() -> SubSystemConstituent:
    """A resistor carrying a port no equation reads, so a caller can name more ports than are needed."""

    @system
    class SensedResistor(metaclass=Interface):
        v: float = field()
        i: float = field()
        r: float = field()
        temperature: float = field()

    @equation
    def ohms_law(v: float, i: float, r: float) -> bool:
        return v == i * r

    return SubSystemConstituent(name="resistor", implementation=Implementation(SensedResistor, [ohms_law]))


@pytest.fixture
def board(sub_system: SubSystemConstituent) -> Implementation:
    """A parent whose only component is ``sub_system``, so its own ports stay unconnected."""

    @system
    class Board(metaclass=Interface):
        supply_voltage: float = field()

    return Implementation(Board, [sub_system])


@pytest.fixture
def wired_board() -> Implementation:
    """A parent whose own port is tied to a child port by an equation taking the child's interface."""

    @system
    class Resistor(metaclass=Interface):
        v: float = field()
        i: float = field()
        r: float = field()

    @equation
    def ohms_law(v: float, i: float, r: float) -> bool:
        return v == i * r

    @system
    class Board(metaclass=Interface):
        supply_voltage: float = field()

    @equation
    def supply_is_across_the_resistor(supply_voltage: float, resistor: Resistor) -> bool:
        return supply_voltage == resistor.v

    resistor = SubSystemConstituent(name="resistor", implementation=Implementation(Resistor, [ohms_law]))
    return Implementation(Board, [supply_is_across_the_resistor, resistor])


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


def test_a_request_names_its_ports_in_child_local_terms(sub_system: SubSystemConstituent) -> None:
    request = sub_system.request_for("resistor.v", PYTHON_BACKEND, within={"resistor.i", "resistor.r"})

    assert request == EvaluationRequest(output="v", available=frozenset({"i", "r"}))


def test_a_request_asks_only_for_the_ports_the_child_uses(
    spare_port_sub_system: SubSystemConstituent,
) -> None:
    """Given a port no equation reads, the child leaves it out of the request."""
    request = spare_port_sub_system.request_for(
        "resistor.v",
        PYTHON_BACKEND,
        within={"resistor.i", "resistor.r", "resistor.temperature"},
    )

    assert request is not None
    assert request.available == frozenset({"i", "r"})


def test_an_unrestricted_request_may_use_every_port_but_the_output(
    spare_port_sub_system: SubSystemConstituent,
) -> None:
    """Without ``within`` the child is free to route from any of its own ports."""
    request = spare_port_sub_system.request_for("resistor.v", PYTHON_BACKEND)

    assert request == EvaluationRequest(output="v", available=frozenset({"i", "r"}))


def test_there_is_no_request_without_an_internal_route(sealed_sub_system: SubSystemConstituent) -> None:
    assert sealed_sub_system.request_for("sealed.a", PYTHON_BACKEND, within={"sealed.b"}) is None


def test_there_is_no_request_when_the_restriction_falls_short(sub_system: SubSystemConstituent) -> None:
    assert sub_system.request_for("resistor.v", PYTHON_BACKEND, within={"resistor.i"}) is None


def test_repr_names_the_reference_and_its_interface(sub_system: SubSystemConstituent) -> None:
    assert repr(sub_system) == "SubSystemConstituent(name='resistor', interface='Resistor')"


def test_a_parent_solves_a_child_port_through_the_sub_system(board: Implementation) -> None:
    result = board.evaluate({"resistor.i": CURRENT, "resistor.r": RESISTANCE}, "resistor.v", PYTHON_BACKEND)

    assert result == VOLTAGE


def test_a_parent_solves_a_child_port_in_the_inward_direction(board: Implementation) -> None:
    result = board.evaluate({"resistor.v": VOLTAGE, "resistor.r": RESISTANCE}, "resistor.i", PYTHON_BACKEND)

    assert result == CURRENT


def test_the_realized_callable_speaks_mangled_names(board: Implementation) -> None:
    """The compiled callable is bound by keyword, so it answers in the names it was compiled with."""
    request = EvaluationRequest(output="resistor.v", available=frozenset({"resistor.i", "resistor.r"}))

    realized = board.get_evaluation(request, PYTHON_BACKEND)

    assert realized.parameters == ("resistor__i", "resistor__r")
    assert realized.evaluate({"resistor__i": CURRENT, "resistor__r": RESISTANCE}) == VOLTAGE


def test_nothing_a_backend_compiles_carries_a_dotted_name(board: Implementation) -> None:
    case = board.analyser_for(PYTHON_BACKEND).analyse({"resistor.i", "resistor.r"}, "resistor.v")

    assert "." not in case.name
    assert "." not in case.return_name
    assert not any("." in name for name in case.parameters)
    assert not any("." in name for intermediate in case.intermediates for name in intermediate.parameters)


def test_a_parent_reuses_one_cache_entry_across_repeated_evaluations(board: Implementation) -> None:
    backend = PYTHON_BACKEND

    board.evaluate({"resistor.i": CURRENT, "resistor.r": RESISTANCE}, "resistor.v", backend)
    board.evaluate({"resistor.i": CURRENT + 1, "resistor.r": RESISTANCE + 1}, "resistor.v", backend)

    assert len(board._evaluation_cache) == 1  # noqa: SLF001


def test_two_parents_can_share_one_sub_system(sub_system: SubSystemConstituent) -> None:
    """A parent relabels the child's callable to its own names; the child's cached one is not its to edit."""

    @system
    class Rig(metaclass=Interface):
        supply_voltage: float = field()

    backend = PYTHON_BACKEND
    inputs: dict[str, object] = {"resistor.i": CURRENT, "resistor.r": RESISTANCE}

    first = Implementation(Rig, [sub_system]).evaluate(inputs, "resistor.v", backend)
    second = Implementation(Rig, [sub_system]).evaluate(inputs, "resistor.v", backend)

    assert first == second == VOLTAGE


def test_a_parent_cannot_solve_a_child_port_the_child_has_no_route_to(
    sealed_sub_system: SubSystemConstituent,
) -> None:
    @system
    class Rack(metaclass=Interface):
        supply_voltage: float = field()

    rack = Implementation(Rack, [sealed_sub_system])

    with pytest.raises(BiltSystemAnalysisError, match="Cannot compute"):
        rack.evaluate({"sealed.b": 1.0}, "sealed.a", PYTHON_BACKEND)


def test_a_sub_system_name_carrying_the_mangled_separator_is_rejected() -> None:
    """``foo__bar.v`` and ``foo.bar.v`` would otherwise both realize as ``foo__bar__v``."""
    with pytest.raises(BiltNameError, match="'foo__bar' may not contain"):
        SubSystemConstituent(name="foo__bar", implementation=_resistor_implementation())


def test_validate_name_accepts_a_name_without_the_separator() -> None:
    assert validate_name("supply_voltage") is None


@pytest.mark.parametrize("name", ("foo__bar", "__leading", "trailing__", "a__b__c"))
def test_validate_name_rejects_a_name_carrying_the_mangled_separator(name: str) -> None:
    with pytest.raises(BiltNameError, match="may not contain"):
        validate_name(name)


def test_a_qualified_name_is_unambiguous_once_no_segment_carries_the_separator() -> None:
    """The rule exists so splitting a flattened name back into segments is well defined."""
    flattened = QualifiedPort("outer", "inner").dotted.replace(DOTTED_SEPARATOR, MANGLED_SEPARATOR)

    assert flattened.split(MANGLED_SEPARATOR) == ["outer", "inner"]

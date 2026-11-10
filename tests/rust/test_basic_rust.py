from bilt._native import Atom, AtomType, Tree


def test_atom_can_be_instantiated() -> None:
    atom_type = AtomType("symbol")
    atom = Atom(atom_type, "x")

    assert atom.atom_type.name == "symbol"
    assert atom.value == "x"


def test_tree_can_be_instantiated() -> None:
    atom_type = AtomType("symbol")
    leaf = Tree(Atom(atom_type, "x"), [])
    tree = Tree(Atom(atom_type, "add"), [leaf, leaf])

    assert leaf.is_leaf()
    assert tree.is_binary()
    assert tree.head.value == "add"
    left = tree.left()
    assert left is not None
    assert left.head.value == "x"

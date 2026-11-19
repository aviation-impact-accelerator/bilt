"""Proposal: choosing implementations, including those of children, when evaluating.

Today an outer `Implementation` fixes its children's implementations when it is
registered (`internal_implementations=`), so an outer model can't be reused with
different inner models. This proposal removes `internal_implementations`: every
implementation is chosen when evaluating, with `with`, which already chooses
implementations at the top level.

**`with` reaches inside systems (change in meaning).** An implementation knows which
interface it implements, so naming it is enough. `with thermal_resistor:` applies to every
`Resistor` anywhere in the model tree, children included, not only to top-level
instances:

    with ideal_wheatstone_bridge, thermal_resistor, ideal_voltage_source, ideal_ground:
        bridge.galvanometer.current

**`Scope(owner, *choices, **children)` (type).** Choices made inside an owner:

- `owner` is an implementation ("this bridge model, with these parts") or an interface
  ("whatever bridge model, with these parts");
- positional `choices` apply by interface to everything below the owner;
- keyword `children` apply to a named child, and take an implementation or another
  `Scope`, so choices nest the way the model does.

```python
# Within the bridge: the galvanometer is thermal, every other resistor ideal.
with Scope(ideal_wheatstone_bridge, ideal_resistor, galvanometer=thermal_resistor), ideal_voltage_source, ideal_ground:
    ...

# Whatever bridge implementation is used, its galvanometer is thermal.
with Scope(WheatstoneBridge, galvanometer=thermal_resistor), ...:
    ...

# Variants build on each other.
standard = Scope(ideal_wheatstone_bridge, ideal_resistor, ideal_voltage_source, ideal_ground)
variant = Scope(standard, galvanometer=thermal_resistor)
```

A `Scope` can be used wherever an implementation can. It is a class rather than a
method on implementations or systems, so that it can't clash with a port of the same
name (the reason `Implementation` is a class).

**Resolution.** For every system instance, the most specific choice wins: a nearer scope
in the model tree beats a further one; at the same scope, a named child beats a choice
by interface, and an implementation owner beats an interface owner. Two choices for one
interface in the same place are an error. An instance with nothing chosen has no
relations: it is a record of ports, which is correct for `Point` or `Pin`.

Names are checked when the `Scope` is built: with an implementation owner, any child
name, private parts included; with an interface owner, public fields only, since private
parts differ between implementations.

**Later additions** (not part of this proposal yet): defaults and fixed choices carried
by an implementation, as `Implementation(..., defaults=Scope(...), fixed=Scope(...))`. A
fixed choice is never overridden; a default is overridden by any choice at least as
specific.
"""

import typing

from bilt_proposals._errors import not_implemented

if typing.TYPE_CHECKING:
    from bilt import Implementation, Interface

    type _Choice = Implementation | Scope


class Scope:
    """An implementation, or an interface, together with the choices made inside it."""

    def __init__(self, owner: Implementation | Interface | Scope, /, *choices: _Choice, **children: _Choice) -> None:
        """Choose implementations inside `owner`: by interface (`choices`) or by child name (`children`)."""
        del owner, choices, children
        not_implemented("Scope")

    def __enter__(self) -> typing.Self:
        """Make these choices for the evaluations inside the `with` block."""
        not_implemented("Scope")

    def __exit__(self, *_: object) -> None:
        """Undo these choices."""
        not_implemented("Scope")

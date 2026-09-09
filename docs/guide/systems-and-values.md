## Systems and values

A Bilt model starts with a **system**.

A system describes the values that make up the boundary of a model. It tells Bilt what exists, but not yet how those values are calculated.

Here is a small example using Newtons second law, `F = m*a`.

```python
from bilt import Interface, field, system


@system
class Demo(metaclass=Interface):
    force: float = field()
    mass: float = field()
    acceleration: float = field()
```

There are three Bilt-specific pieces here.

`@system` tells Bilt that the class describes a system.

`Interface` is the metaclass used for Bilt system interfaces. You do not need to understand Python metaclasses to use Bilt; for now, think of this as part of the standard way a system is declared.

Finally, `field()` declares a value on the system boundary:

```python
force: float = field()
```

Here, `force` is the name of the value and `float` is its type.

So our system has three values:

```text
Demo
├── force: float
├── mass: float
└── acceleration: float
```

Notice that we still haven't said how any of them are related. Declaring `acceleration` means that `acceleration` is part of the system interface; it does not mean that Bilt knows how to calculate it.

We can create an instance by supplying values that we already know:

```python
mass = 1.0
acceleration = 2.0

demo = Demo(mass=mass, acceleration=acceleration)
```

Here, `mass` and `acceleration` are known, while `force` has not been supplied.

Whether Bilt can obtain `force` depends on the relationships provided by the system's implementation. We'll introduce those relationships in the next chapter.

### Values aren't permanently inputs or outputs

It is tempting to think of `mass` and `acceleration` as inputs and `force` as an output, but Bilt does not assign those roles permanently.

A system simply declares values.

Which values are known and which value is requested depends on the particular evaluation.

For one analysis we might know:

```text
mass, force
```

and ask for:

```text
acceleration
```

In another analysis, the available values could be different.

This becomes particularly useful once equations are involved, because the same mathematical relationship may be usable in more than one direction.

For now, the important distinction is simpler:

> A system defines the boundary of a model. It does not define the computation.

In the next chapter, we'll introduce **acausal modelling** and **equations**, before learning about the **implementations** that link these to a system.

---

[Index](../index.md) · [Equations and acausal modelling →](equations-and-acausal-modelling.md)

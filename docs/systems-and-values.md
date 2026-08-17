## Systems and values

A Bilt model starts with a **system**.

A system describes the values that make up the boundary of a model. It tells Bilt what exists, but not yet how those values are calculated.

Here is a small example:

```python
from bilt import Interface, field, system


@system
class Demo(metaclass=Interface):
    a: float = field()
    b: float = field()
    c: float = field()
```

There are three Bilt-specific pieces here.

`@system` tells Bilt that the class describes a system.

`Interface` is the metaclass used for Bilt system interfaces. You do not need to understand Python metaclasses to use Bilt; for now, think of this as part of the standard way a system is declared.

Finally, `field()` declares a value on the system boundary:

```python
a: float = field()
```

Here, `a` is the name of the value and `float` is its type.

So our system has three values:

```text
Demo
├── a: float
├── b: float
└── c: float
```

Notice that we still haven't said how any of them are related. Declaring `c` means that `c` is part of the system interface; it does not mean that Bilt knows how to calculate it.

We can create an instance by supplying values that we already know:

```python
a = 1.0
b = 2.0

demo = Demo(a=a, b=b)
```

Here, `a` and `b` are known, while `c` has not been supplied.

Whether Bilt can obtain `c` depends on the relationships provided by the system's implementation. We'll introduce those relationships in the next chapter.

### Values aren't permanently inputs or outputs

It is tempting to think of `a` and `b` as inputs and `c` as an output, but Bilt does not assign those roles permanently.

A system simply declares values.

Which values are known and which value is requested depends on the particular evaluation.

For one analysis we might know:

```text
a, b
```

and ask for:

```text
c
```

In another analysis, the available values could be different.

This becomes particularly useful once equations are involved, because the same mathematical relationship may be usable in more than one direction.

For now, the important distinction is simpler:

> A system defines the boundary of a model. It does not define the computation.

In the next chapter, we'll introduce **transforms** and **implementations** and use them to give a system behaviour.

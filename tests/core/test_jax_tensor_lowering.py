import ast
import inspect
import textwrap
import typing

import numpy as np
import pytest

from bilt._backend._ast import generate_reshaping_assignment
from bilt._backend._jax._backend import JAXBackend
from bilt.core._evaluation_context import EvaluationContext
from bilt.core._interface import Interface
from bilt.core._system._decorator import system
from bilt.core._system._field import field
from bilt.core._system._implementation import Implementation
from bilt.core._tensors._evaluation_domain_analysis import identify_extra_dims
from bilt.core._tensors._lowering_and_lifting import (  # pyright: ignore[reportPrivateUsage]
    _align_extra_dims,
    _transpose_tensor_to_canonical_order,
)
from bilt.core._tensors._reshaping import derive_reshape_slices
from bilt.core._tensors._tensors import Tensor, tensor
from bilt.core._tensors._transform_domain_analysis import (
    _extract_domains_from_transform_signature,  # pyright: ignore[reportPrivateUsage]
)
from bilt.core._transform import transform
from bilt.symbolic._dimensions import dim
from bilt.symbolic._domain_handling import canonicalised, union
from bilt.symbolic._domains import CanonicalisedDomain, Domain, domain

i = typing.Literal["i"]
j = typing.Literal["j"]
k = typing.Literal["k"]
m = typing.Literal["m"]
n = typing.Literal["n"]

dim_i = dim("i")
dim_j = dim("j")
dim_k = dim("k")
dim_m = dim("m")
dim_n = dim("n")


def test_jax_tensor_evaluation() -> None:
    @system
    class Demo(metaclass=Interface):
        a: Tensor[i] = field()
        b: Tensor[j] = field()
        c: Tensor[i, j] = field()
        m: Tensor[None] = field()

        d: Tensor[i, j] = field()

    @transform
    def c(a: Tensor[i], b: Tensor[j]) -> Tensor[i, j]:
        return a * b

    @transform
    def d(c: Tensor[i, j], m: Tensor[None]) -> Tensor[i, j]:
        return m * (c * c)

    basic = Implementation(Demo, [c, d])
    a: Tensor[i] = tensor([12.0, 58.0, 23.0, 91.0, 16.0], over=dim_i)
    b: Tensor[j] = tensor([0.7, 0.2, 0.35], over=dim_j)
    m: Tensor[None] = tensor(5)
    demo = Demo(a=a, b=b, m=m)

    expected = tensor(
        [
            [352.8, 28.8, 88.2],
            [8241.8, 672.8, 2060.45],
            [1296.05, 105.8, 324.0125],
            [20288.45, 1656.2, 5072.1125],
            [627.2, 51.2, 156.8],
        ],
        over=(dim_i, dim_j),
    )

    with basic, JAXBackend():
        result = demo.d

        assert isinstance(result, Tensor)
        assert result.domain == expected.domain
        np.testing.assert_allclose(expected.values, result.values, atol=0.1)


def test_swapped_input_dimensions() -> None:
    @transform
    def c(a: Tensor[i, j], b: Tensor[i, j]) -> Tensor[i, j]:
        return b - a

    @system
    class Demo(metaclass=Interface):
        a: Tensor[i, j] = field()
        b: Tensor[i, j] = field()
        c: Tensor[i, j] = field()

    basic = Implementation(Demo, [c])

    a: Tensor = tensor(
        [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ],
        over=(dim_i, dim_j),
    )

    b: Tensor = tensor(
        [
            [1, 4, 7],
            [2, 5, 8],
            [3, 6, 9],
        ],
        over=(dim_j, dim_i),
    )

    demo = Demo(a=a, b=b)

    expected = tensor(np.zeros((3, 3)), over=(dim_i, dim_j))

    with basic, JAXBackend():
        result = demo.c
        assert expected.domain == result.domain
        np.testing.assert_allclose(expected.values, result.values)


def test_transform_callables_expect_dimensions_in_canonical_order() -> None:
    @transform
    def c(a: Tensor[i, j], b: Tensor[j, i]) -> Tensor[i, j]:
        return b - a

    @system
    class Demo(metaclass=Interface):
        a: Tensor[i, j] = field()
        b: Tensor[i, j] = field()
        c: Tensor[i, j] = field()

    basic = Implementation(Demo, [c])

    a: Tensor = tensor(
        [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ],
        over=(dim_i, dim_j),
    )

    b: Tensor = tensor(
        [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ],
        over=(dim_i, dim_j),
    )

    demo = Demo(a=a, b=b)

    expected = tensor(np.zeros((3, 3)), over=(dim_i, dim_j))

    with basic, JAXBackend():
        result = demo.c
        assert result.domain == expected.domain
        np.testing.assert_allclose(expected.values, result.values)


def test_extract_domains_from_func_def_signature() -> None:
    @transform
    def c(a: Tensor[i], b: Tensor[j]) -> Tensor[i, j]:
        return a * b

    @transform
    def d(c: Tensor[i, j], m: Tensor[None]) -> Tensor[i, j]:
        return m * (c * c)

    c_def = typing.cast(
        "ast.FunctionDef",
        ast.parse(
            textwrap.dedent(
                inspect.getsource(c._function)  # noqa: SLF001 # pyright: ignore[reportPrivateUsage]
            )
        ).body[0],
    )

    d_def = typing.cast(
        "ast.FunctionDef",
        ast.parse(
            textwrap.dedent(
                inspect.getsource(d._function)  # noqa: SLF001 # pyright: ignore[reportPrivateUsage]
            )
        ).body[0],
    )

    c_extracted = _extract_domains_from_transform_signature(c_def)
    c_expected = {
        "a": domain(dim_i),
        "b": domain(dim_j),
        "return": domain(dim_i, dim_j),
    }

    assert c_extracted == c_expected

    d_extracted = _extract_domains_from_transform_signature(d_def)
    d_expected = {
        "c": domain(dim_i, dim_j),
        "m": Domain.scalar(),
        "return": domain(dim_i, dim_j),
    }

    assert d_extracted == d_expected


def test_domain_union() -> None:
    domain1 = Domain((dim_i, dim_j))
    domain2 = Domain((dim_k, dim_j))

    union_ = union(domain1, domain2)

    assert union_ == Domain((dim_i, dim_j, dim_k))


def test_derive_reshape_slices() -> None:
    domain1 = CanonicalisedDomain(domain(i))
    domain2 = CanonicalisedDomain(domain(j))
    domain3 = CanonicalisedDomain(domain(i, j))
    domain4 = CanonicalisedDomain(domain(i, j, k))
    domain5 = CanonicalisedDomain(domain(j, k))
    domain6 = CanonicalisedDomain(domain(i, k))

    assert derive_reshape_slices(domain1, domain3) == (slice(None), None)
    assert derive_reshape_slices(domain2, domain3) == (None, slice(None))
    assert derive_reshape_slices(domain3, domain4) == (slice(None), slice(None), None)
    assert derive_reshape_slices(domain5, domain4) == (None, slice(None), slice(None))
    assert derive_reshape_slices(domain6, domain4) == (slice(None), None, slice(None))
    assert derive_reshape_slices(domain4, domain4) == (slice(None),) * 3


def test_derive_reshape_slices_in_different_order_errors() -> None:
    domain1 = CanonicalisedDomain(domain(dim_i, dim_j))
    domain3 = CanonicalisedDomain(domain(dim_j, dim_i))

    with pytest.raises(RuntimeError):
        _ = derive_reshape_slices(domain1, domain3)


def test_make_reshaping_assignment() -> None:
    reshape1 = (slice(None), None)
    reshape2 = (None, slice(None))
    reshape3 = (slice(None), slice(None), None)
    reshape4 = (None, slice(None), slice(None))
    reshape5 = (slice(None), None, slice(None))
    reshape6 = (slice(None),) * 3

    assignment1 = generate_reshaping_assignment("a", reshape1)
    assignment2 = generate_reshaping_assignment("a", reshape2)
    assignment3 = generate_reshaping_assignment("a", reshape3)
    assignment4 = generate_reshaping_assignment("a", reshape4)
    assignment5 = generate_reshaping_assignment("a", reshape5)
    assignment6 = generate_reshaping_assignment("a", reshape6)

    ast.fix_missing_locations(assignment1)
    ast.fix_missing_locations(assignment2)
    ast.fix_missing_locations(assignment3)
    ast.fix_missing_locations(assignment4)
    ast.fix_missing_locations(assignment5)
    ast.fix_missing_locations(assignment6)

    assert ast.unparse(assignment1) == "a = a[:, None]"
    assert ast.unparse(assignment2) == "a = a[None, :]"
    assert ast.unparse(assignment3) == "a = a[:, :, None]"
    assert ast.unparse(assignment4) == "a = a[None, :, :]"
    assert ast.unparse(assignment5) == "a = a[:, None, :]"
    assert ast.unparse(assignment6) == "a = a[:, :, :]"


def test_identify_extra_dims() -> None:
    assert identify_extra_dims(Domain.scalar(), Domain.scalar()) == Domain(())
    assert identify_extra_dims(domain(dim_i, dim_j, dim_k), domain(dim_i, dim_j)) == Domain((dim_k,))
    assert identify_extra_dims(domain(dim_i, dim_j), domain(dim_i, dim_j)) == Domain(())
    assert identify_extra_dims(domain(dim_k, dim_i), domain(dim_i, dim_j)) == Domain((dim_k,))


def test_transpose_tensor_to_canonical_order() -> None:
    canonical_order = CanonicalisedDomain(domain(dim_i, dim_j, dim_k))

    a = tensor(
        [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ],
        over=(dim_i, dim_j),
    )

    a_canonical = _transpose_tensor_to_canonical_order(a, canonical_order)

    np.testing.assert_allclose(
        a_canonical.values,
        [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ],
    )

    assert a_canonical.domain.dims == (dim_i, dim_j)

    b = tensor([[1, 4, 7], [2, 5, 8], [3, 6, 9]], over=(dim_j, dim_i))

    assert b.domain.dims == (dim_j, dim_i)

    b_canonical = _transpose_tensor_to_canonical_order(b, canonical_order)

    np.testing.assert_allclose(
        b_canonical.values,
        [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ],
    )

    assert b_canonical.domain.dims == (dim_i, dim_j)


def test_canonicalised() -> None:
    with EvaluationContext() as evaluation:
        a = tensor(
            [
                [1, 2],
                [3, 4],
            ],
            over=(dim_i, dim_j),
        )

        b = tensor(
            [
                [1, 3],
                [2, 4],
            ],
            over=(dim_j, dim_i),
        )

        evaluation.domains.bind_all([a, b])

        assert evaluation.domains.get_canonical_domain() == CanonicalisedDomain(domain(dim_i, dim_j))

        assert canonicalised(domain(dim_i, dim_j)) == CanonicalisedDomain(domain(dim_i, dim_j))
        assert canonicalised(domain(dim_j, dim_i)) == CanonicalisedDomain(domain(dim_i, dim_j))


def test_scalar_transform_still_works() -> None:
    @system
    class Demo(metaclass=Interface):
        a: float = field()
        b: float = field()
        c: float = field()

    @transform
    def c(a: float, b: float) -> float:
        return a * b

    basic = Implementation(Demo, [c])
    demo = Demo(a=5.0, b=3.0)

    expected = 15.0

    with basic, JAXBackend():
        result = demo.c

        assert isinstance(result, float)
        assert result == expected


def test_extra_dims_in_scalar_computation_single_dim() -> None:
    @system
    class Demo(metaclass=Interface):
        a: Tensor[None] = field()
        b: float = field()
        c: float = field()
        d: float = field()
        e: Tensor = field()

    @transform
    def c(a: Tensor[None], b: Tensor[None]) -> Tensor[None]:
        return a * b

    @transform
    def d(c: Tensor[None], e: Tensor[None]) -> Tensor[None]:
        return c + e

    basic = Implementation(Demo, [c, d])
    a = tensor([1.0, 2.0, 3.0], over=(dim_i,))
    b = 3.0
    e = tensor([2.0, 4.0, 6.0], over=(dim_i,))
    demo = Demo(a=a, b=b, e=e)

    with basic, JAXBackend():
        result = demo.d

        assert isinstance(result, Tensor)
        assert result.domain == domain(dim_i)
        np.testing.assert_allclose(result.values, np.array([5.0, 10.0, 15.0]))


def test_extra_dims_in_scalar_computation_two_dims() -> None:
    @system
    class Demo(metaclass=Interface):
        a: Tensor = field()
        b: float = field()
        c: Tensor = field()
        d: Tensor = field()
        e: Tensor = field()

    @transform
    def c(a: Tensor[None], b: Tensor[None]) -> Tensor[None]:
        return a * b

    @transform
    def d(c: Tensor[None], e: Tensor[None]) -> Tensor[None]:
        return c + e

    with EvaluationContext() as evaluation:
        basic = Implementation(Demo, [c, d])
        a = tensor([1.0, 2.0, 3.0], over=(dim_i,))
        b = 3
        e = tensor([2.0, 4.0, 6.0], over=(dim_j,))
        demo = Demo(a=a, b=b, e=e)

        evaluation.domains.bind_all([a, b, e])
        assert evaluation.domains.get_canonical_domain() == CanonicalisedDomain(domain(dim_i, dim_j))

        with basic, JAXBackend():
            result = demo.d

            assert isinstance(result, Tensor)
            assert result.domain == domain(dim_i, dim_j)
            np.testing.assert_allclose(
                result.values,
                np.array(
                    [
                        [5.0, 7.0, 9.0],
                        [8.0, 10.0, 12.0],
                        [11.0, 13.0, 15.0],
                    ]
                ),
            )


def test_align_extra_dims() -> None:
    with EvaluationContext() as evaluation:
        a = tensor(
            [
                [
                    [1, 2],
                    [3, 4],
                ],
                [
                    [5, 6],
                    [7, 8],
                ],
            ],
            over=(dim_i, dim_j, dim_m),
        )

        b = tensor(
            [
                [
                    [3, 6],
                    [2, 5],
                ],
                [
                    [9, 1],
                    [4, 2],
                ],
            ],
            over=(dim_i, dim_j, dim_n),
        )

        c = tensor([1, 2], over=(dim_i,))

        evaluation.domains.bind_all([a, b, c])

        assert evaluation.domains.get_canonical_domain() == CanonicalisedDomain(domain(dim_i, dim_j, dim_m, dim_n))

        extra_dims = CanonicalisedDomain(domain(dim_m, dim_n))

        res_a = _align_extra_dims(a, extra_dims)
        np.testing.assert_allclose(res_a.values, a.values[:, :, :, None])
        assert res_a.domain == domain(dim_i, dim_j, dim_m, dim_n)

        res_b = _align_extra_dims(b, extra_dims)
        np.testing.assert_allclose(res_b.values, b.values[:, :, None, :])
        assert res_b.domain == domain(dim_i, dim_j, dim_m, dim_n)

        res_c = _align_extra_dims(c, extra_dims)
        np.testing.assert_allclose(res_c.values, c.values[:, None, None])
        assert res_c.domain == domain(dim_i, dim_m, dim_n)


def test_one_chained_callable_uses_the_others_extra_dims() -> None:
    component = dim("component")
    time = dim("time")

    @system
    class Demo(metaclass=Interface):
        voltage: Tensor[component] = field()
        current: Tensor[component] = field()
        power: Tensor[component] = field()
        noise: Tensor[time] = field()
        noisy_power: Tensor[time] = field()

    # Component is expected, Time is extra
    @transform
    def power(voltage: Tensor[component], current: Tensor[component]) -> Tensor[component]:
        return voltage * current

    # Time is expected, Component is extra
    @transform
    def noisy_power(power: Tensor[time], noise: Tensor[time]) -> Tensor[time]:
        return power + noise

    with EvaluationContext() as evaluation:
        basic = Implementation(Demo, [power, noisy_power])
        voltage = tensor([5.0, 3.3, 4.5], over=(component,))
        current = tensor([3.0, 3.1, 1.6], over=(component,))
        noise = tensor([0.11, 0.12, 0.09, 0.10, 0.11, 0.12, 0.11], over=(time,))

        evaluation.domains.bind_all([voltage, current, noise])
        assert evaluation.domains.get_canonical_domain() == CanonicalisedDomain(domain(component, time))

        demo = Demo(voltage=voltage, current=current, noise=noise)

        with basic, JAXBackend():
            result = demo.noisy_power

            assert isinstance(result, Tensor)
            assert result.domain == domain(component, time)
            np.testing.assert_allclose(
                result.values,
                np.array(
                    [
                        [15.11, 15.12, 15.09, 15.10, 15.11, 15.12, 15.11],
                        [10.34, 10.35, 10.32, 10.33, 10.34, 10.35, 10.34],
                        [7.31, 7.32, 7.29, 7.30, 7.31, 7.32, 7.31],
                    ]
                ),
            )

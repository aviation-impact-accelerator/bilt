import ast
import inspect
import textwrap
import typing

import numpy as np
import pytest

from bilt._backend._jax._backend import JAXBackend
from bilt.core._evaluation_context import EvaluationContext
from bilt.core._interface import Interface
from bilt.core._system._decorator import system
from bilt.core._system._field import field
from bilt.core._system._implementation import Implementation
#from bilt.core._tensors._tensor_analysis import extract_domains_from_signature
from bilt.core._tensors._tensors import Tensor, tensor
from bilt.core._transform import transform
from bilt.symbolic._dimensions import dim
from bilt.core._tensors._tensor_analysis import *

from bilt.symbolic._domains import Domain, domain

i = dim("i")
j = dim("j")
k = dim("k")
m = dim("m")
n = dim("n")

def test_jax_tensor_evaluation() -> None:
    @system
    class Demo(metaclass=Interface):
        a: Tensor[i] = field()
        b: Tensor[j] = field()
        c: Tensor[i,j] = field()
        m: Tensor[None] = field()

        d: Tensor[i,j] = field()

    @transform
    def c(a: Tensor[i], b: Tensor[j]) -> Tensor[i,j]:
        return a * b

    @transform
    def d(c: Tensor[i,j], m: Tensor[None]) -> Tensor[i,j]:
        return m * (c * c)

    basic = Implementation(Demo, [c, d])
    a: Tensor[i] = tensor([12., 58., 23., 91., 16.], over=i)
    b: Tensor[j] = tensor([0.7, 0.2, 0.35], over=j)
    #m: Tensor[k] = tensor([1, 2, 3, 4], over=k)
    m: Tensor[None] = tensor(5)
    demo = Demo(a=a, b=b, m=m)

    expected = tensor([
        [  352.8   ,    28.8   ,    88.2   ],
        [ 8241.8   ,   672.8   ,  2060.45  ],
        [ 1296.05  ,   105.8   ,   324.0125],
        [20288.45  ,  1656.2   ,  5072.1125],
        [  627.2   ,    51.2   ,   156.8   ]
    ], over=(i,j))

    with basic, JAXBackend():
        result = demo.d

        assert isinstance(result, Tensor)
        assert result.domain == expected.domain
        np.testing.assert_allclose(expected.values, result.values, atol=0.1)

def test_swapped_input_dimensions() -> None:
    @transform
    def c(a: Tensor[i,j], b: Tensor[i,j]) -> Tensor[i,j]:
        return b - a

    @system
    class Demo(metaclass=Interface):
        a: Tensor[i,j] = field()
        b: Tensor[i,j] = field()
        c: Tensor[i,j] = field()

    basic = Implementation(Demo, [c])

    a: Tensor = tensor([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
    ], over=(i,j))

    b: Tensor = tensor([
        [1, 4, 7],
        [2, 5, 8],
        [3, 6, 9],
    ], over=(j,i))

    demo = Demo(a=a, b=b)

    expected = tensor(
        np.zeros((3,3)),
        over=(i,j)
    )

    with basic, JAXBackend():
        result = demo.c
        assert expected.domain == result.domain
        np.testing.assert_allclose(expected.values, result.values)

def test_transform_callables_expect_dimensions_in_canonical_order() -> None:
    @transform
    def c(a: Tensor[i,j], b: Tensor[j,i]) -> Tensor[i,j]:
        return b - a

    @system
    class Demo(metaclass=Interface):
        a: Tensor[i,j] = field()
        b: Tensor[i,j] = field()
        c: Tensor[i,j] = field()

    basic = Implementation(Demo, [c])

    a: Tensor = tensor([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
    ], over=(i,j))

    b: Tensor = tensor([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
    ], over=(i,j))

    demo = Demo(a=a, b=b)

    expected = tensor(
        np.zeros((3,3)),
        over=(i,j)
    )

    with basic, JAXBackend():
        result = demo.c
        assert result.domain == expected.domain
        np.testing.assert_allclose(expected.values, result.values)

def test_extract_domains_from_func_def_signature():
    @transform
    def c(a: Tensor[i], b: Tensor[j]) -> Tensor[i,j]:
        return a * b

    @transform
    def d(c: Tensor[i,j], m: Tensor[None]) -> Tensor[i,j]:
        return m * (c * c)

    c_def = typing.cast(ast.FunctionDef,
                        ast.parse(
                            textwrap.dedent(
                                inspect.getsource(c._function)  # pyright: ignore[reportPrivateUsage]
                            )
                        ).body[0])
    
    d_def = typing.cast(ast.FunctionDef,
                        ast.parse(
                            textwrap.dedent(
                                inspect.getsource(d._function)  # pyright: ignore[reportPrivateUsage]
                            )
                        ).body[0])

    c_extracted = extract_domains_from_func_def_signature(c_def)
    c_expected = {
        "a": domain(i),
        "b": domain(j),
        "return": domain(i,j),
    }

    assert c_extracted == c_expected

    d_extracted = extract_domains_from_func_def_signature(d_def)
    d_expected = {
        "c": domain(i,j),
        "m": Domain.scalar(),
        "return": domain(i,j),
    }

    assert d_extracted == d_expected

def test_domain_union():
    domain1 = Domain((i, j))
    domain2 = Domain((k, j))

    union_ = union(domain1, domain2)

    assert union_ == Domain((i, j, k))

def test_derive_reshape_slices():
    domain1 = domain(i)
    domain2 = domain(j)
    domain3 = domain(i, j)
    domain4 = domain(i, j, k)
    domain5 = domain(j, k)
    domain6 = domain(i, k)

    assert derive_reshape_slices(domain1, domain3) == (slice(None), None)
    assert derive_reshape_slices(domain2, domain3) == (None, slice(None))
    assert derive_reshape_slices(domain3, domain4) == (slice(None), slice(None), None)
    assert derive_reshape_slices(domain5, domain4) == (None, slice(None), slice(None))
    assert derive_reshape_slices(domain6, domain4) == (slice(None), None, slice(None))
    assert derive_reshape_slices(domain4, domain4) == (slice(None),)*3

def test_derive_reshape_slices_in_different_order_errors():
    domain1 = domain(i, j)
    domain3 = domain(j, i)

    with pytest.raises(RuntimeError):
        _ = derive_reshape_slices(domain1, domain3)

def test_make_reshaping_assignment():
    reshape1 = (slice(None), None)
    reshape2 = (None, slice(None))
    reshape3 = (slice(None), slice(None), None)
    reshape4 = (None, slice(None), slice(None))
    reshape5 = (slice(None), None, slice(None))
    reshape6 = (slice(None),)*3

    assignment1 = make_reshaping_assignment("a", reshape1)
    assignment2 = make_reshaping_assignment("a", reshape2)
    assignment3 = make_reshaping_assignment("a", reshape3)
    assignment4 = make_reshaping_assignment("a", reshape4)
    assignment5 = make_reshaping_assignment("a", reshape5)
    assignment6 = make_reshaping_assignment("a", reshape6)

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

def test_domain_to_tensor_type_string():
    assert domain_to_tensor_type_string(domain(i)) == "Tensor[i]"
    assert domain_to_tensor_type_string(domain(j)) == "Tensor[j]"
    assert domain_to_tensor_type_string(domain(i,j)) == "Tensor[i,j]"
    assert domain_to_tensor_type_string(domain(j,i)) == "Tensor[j,i]"
    assert domain_to_tensor_type_string(Domain.scalar()) == "Tensor[None]"

def test_convert_tensors_to_raw_arrays_in_inputs():
    inputs: dict[str, object] = {
        "a": 5,
        "b": 2,
        "c": tensor([1, 2, 3], over=i),
        "d": 4,
        "e": tensor([[1, 2, 3], [4, 5, 6]], over=(j,k)),
    }

    result = convert_any_tensors_to_raw_arrays(inputs)
    expected = {
        "a": 5,
        "b": 2,
        "c": np.array([1, 2, 3]),
        "d": 4,
        "e": np.array([[1, 2, 3], [4, 5, 6]])
    }

    np.testing.assert_equal(result, expected)

def test_identify_extra_dims():
    assert identify_extra_dims(Domain.scalar(), Domain.scalar()) == tuple()
    assert identify_extra_dims(domain(i, j, k), domain(i, j)) == (k,)
    assert identify_extra_dims(domain(i, j), domain(i, j)) == tuple()
    assert identify_extra_dims(domain(k, i), domain(i, j)) == (k,)

def test_transpose_tensor_to_canonical_order():
    canonical_order = [i, j, k]

    a = tensor([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
    ], over=(i,j))

    a_canonical = transpose_tensor_to_canonical_order(a, canonical_order)

    np.testing.assert_allclose(a_canonical.values, [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
    ])

    assert a_canonical.domain.dims == (i,j)

    b = tensor([
        [1, 4, 7],
        [2, 5, 8],
        [3, 6, 9]
    ], over=(j,i))

    assert b.domain.dims == (j,i)

    b_canonical = transpose_tensor_to_canonical_order(b, canonical_order)

    np.testing.assert_allclose(b_canonical.values, [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
    ])

    assert b_canonical.domain.dims == (i,j)

def test_canonicalised():
    with EvaluationContext() as evaluation:
        a = tensor([
            [1,2],
            [3,4],
        ], over=(i,j))

        b = tensor([
            [1,3],
            [2,4],
        ], over=(j,i))

        evaluation.domains.bind_all([a, b])

        assert list(evaluation.domains._bindings.keys()) == [i,j]

        assert canonicalised(domain(i,j)) == domain(i,j)
        assert canonicalised(domain(j,i)) == domain(i,j)

def test_scalar_transform_still_works():
    @system
    class Demo(metaclass=Interface):
        a: float = field()
        b: float = field()
        c: float = field()

    @transform
    def c(a: float, b: float) -> float:
        return a * b

    basic = Implementation(Demo, [c])
    demo = Demo(a=5., b=3.)

    with basic, JAXBackend():
        result = demo.c

        assert isinstance(result, float)
        assert result == 15.

def test_extra_dims_in_scalar_computation_single_dim():
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
    a = tensor([1., 2., 3.], over=(i,))
    b = 3.
    e = tensor([2., 4., 6.], over=(i,))
    demo = Demo(a=a, b=b, e=e)

    with basic, JAXBackend():
        result = demo.d

        assert isinstance(result, Tensor)
        assert result.domain == domain(i)
        np.testing.assert_allclose(result.values, np.array([5., 10., 15.]))

def test_extra_dims_in_scalar_computation_two_dims():
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
        a = tensor([1., 2., 3.], over=(i,))
        b = 3
        e = tensor([2., 4., 6.], over=(j,))
        demo = Demo(a=a, b=b, e=e)

        evaluation.domains.bind_all([a, b, e])
        assert list(evaluation.domains._bindings.keys()) == [i,j]

        with basic, JAXBackend():
            result = demo.d

            assert isinstance(result, Tensor)
            assert result.domain == domain(i,j)
            np.testing.assert_allclose(result.values, np.array([
                [5., 7., 9.],
                [8., 10., 12.],
                [11., 13., 15.],
            ]))

def test_align_extra_dims():
    with EvaluationContext() as evaluation:
        a = tensor([
            [
                [1, 2],
                [3, 4],
            ],
            [
                [5, 6],
                [7, 8],
            ]
        ], over=(i,j,m))

        b = tensor([
            [
                [3, 6],
                [2, 5],
            ],
            [
                [9, 1],
                [4, 2],
            ]
        ], over=(i,j,n))

        c = tensor([1, 2], over=(i,))

        evaluation.domains.bind_all([a, b, c])

        assert list(evaluation.domains._bindings.keys()) == [i, j, m, n]

        extra_dims = [m, n]

        res_a = align_extra_dims(a, extra_dims)
        np.testing.assert_allclose(res_a.values, a.values[:,:,:,None])
        assert res_a.domain == domain(i, j, m, n)

        res_b = align_extra_dims(b, extra_dims)
        np.testing.assert_allclose(res_b.values, b.values[:,:,None,:])
        assert res_b.domain == domain(i, j, m, n)

        res_c = align_extra_dims(c, extra_dims)
        np.testing.assert_allclose(res_c.values, c.values[:,None,None])
        assert res_c.domain == domain(i, m, n)

def test_one_chained_callable_uses_the_others_extra_dims():
    Component = dim("Component")
    Time = dim("Time")

    @system
    class Demo(metaclass=Interface):
        voltage: Tensor[Component] = field()
        current: Tensor[Component] = field()
        power: Tensor[Component] = field()
        noise: Tensor[Time] = field()
        noisy_power: Tensor[Time] = field()

    # Component is expected, Time is extra
    @transform
    def power(voltage: Tensor[Component], current: Tensor[Component]) -> Tensor[Component]:
        return voltage * current

    # Time is expected, Component is extra
    @transform
    def noisy_power(power: Tensor[Time], noise: Tensor[Time]) -> Tensor[Time]:
        return power + noise

    with EvaluationContext() as evaluation:
        basic = Implementation(Demo, [power, noisy_power])
        voltage = tensor([5., 3.3, 4.5], over=(Component,))
        current = tensor([3., 3.1, 1.6], over=(Component,))
        noise = tensor([0.11, 0.12, 0.09, 0.10, 0.11, 0.12, 0.11], over=(Time,))

        evaluation.domains.bind_all([voltage, current, noise])
        assert list(evaluation.domains._bindings.keys()) == [Component,Time]

        demo = Demo(voltage=voltage, current=current, noise=noise)

        with basic, JAXBackend():
            result = demo.noisy_power

            assert isinstance(result, Tensor)
            assert result.domain == domain(Component,Time)
            np.testing.assert_allclose(result.values, np.array([
                [15.11, 15.12, 15.09, 15.10, 15.11, 15.12, 15.11],
                [10.34, 10.35, 10.32, 10.33, 10.34, 10.35, 10.34],
                [ 7.31,  7.32,  7.29,  7.30,  7.31,  7.32,  7.31],
            ]))

def test_domain_registry_ordering_is_stable():
    with EvaluationContext() as evaluation:
        a = tensor([1., 2., 3.], over=(i,))
        b = 3
        e = tensor([2., 4., 6.], over=(j,))

        evaluation.domains.bind_all([a, b, e])
        assert list(evaluation.domains._bindings.keys()) == [i,j]
import ast

from bilt._backend._callable import BackendCallable
from bilt._backend._case import BackendCase
from bilt._backend._jax._tensor_handling import IntermediateAnalysis, analyse_intermediates
from bilt.core._tensors._tensor_analysis import SLICE_COLON, SLICE_NEWAXIS, ReshapeSlices


def generate_function_args_ast(case: BackendCase) -> ast.arguments:
    return ast.arguments(
        posonlyargs=[],
        args=[ast.arg(arg=name) for name in case.parameters.keys()],
        kwonlyargs=[],
        kw_defaults=[],
        defaults=[],
    )

def generate_intermediate_calls_ast(case: BackendCase) -> list[ast.stmt]:
    function_body: list[ast.stmt] = []

    intermediates_analysis = analyse_intermediates(case)

    for callable in case.intermediates.values():
        function_body.append(_generate_intermediate_call_ast(callable, intermediates_analysis.intermediates[callable.name]))

    return function_body

def _generate_intermediate_call_ast(callable: BackendCallable, call_arg_analysis: IntermediateAnalysis) -> ast.stmt: 
    call_args = generate_call_args(callable, call_arg_analysis)

    return ast.Assign(
        targets=[ast.Name(id=f"{callable.name}_", ctx=ast.Store())],
        value=ast.Call(
            func=ast.Name(id=callable.name, ctx=ast.Load()),
            args=call_args,
            keywords=[],
        )
    )

def generate_call_args(callable: BackendCallable, intermediate_analysis: IntermediateAnalysis) -> list[ast.expr]:
    call_args: list[ast.expr] = []

    for expected_param in callable.expected_parameters.values():
        analysed_param = intermediate_analysis.parameters[expected_param.name]
        actual_name = analysed_param.var.actual_name

        if analysed_param.reshape_slices is not None:
            call_args.append(
                generate_slice_ast(actual_name, analysed_param.reshape_slices)
            )
        else:
            call_args.append(ast.Name(id=actual_name, ctx=ast.Load()))

    return call_args
        

def generate_function_return(case: BackendCase) -> ast.Return:
    return ast.Return(
        value=ast.Name(
            id=f"{case.returns.name}_",
            ctx=ast.Load()
        )
    )

def generate_chain_case_ast(name: str, function_args: ast.arguments, function_intermediate_calls: list[ast.stmt], function_return: ast.Return) -> ast.FunctionDef:
    return ast.FunctionDef(
        name=name,
        args=function_args,
        body=function_intermediate_calls + [function_return],
        decorator_list=[],
    )

def generate_slice_ast(param_name: str, reshape_slices: ReshapeSlices) -> ast.Subscript:
    elts: list[ast.expr] = []
    
    for slice in reshape_slices:
        if slice == SLICE_COLON:
            elts.append(ast.Slice(lower=None, upper=None, step=None))
        elif slice == SLICE_NEWAXIS:
            elts.append(ast.Constant(value=None))
        else:
            assert False

    return ast.Subscript(
        value=ast.Name(id=param_name, ctx=ast.Load()),
        slice=ast.Tuple(elts=elts, ctx=ast.Load()),
        ctx=ast.Load(),
    )


def generate_reshaping_assignment(param_name: str, reshape_slices: ReshapeSlices) -> ast.Assign:
    """Generates an AST Assign node for: target_name = target_name[reshape_slices]"""
    return ast.Assign(
        targets=[ast.Name(id=param_name, ctx=ast.Store())],
        value=generate_slice_ast(param_name, reshape_slices),
    )
import ast
import dataclasses
import typing

from bilt.core._tensors._errors import UnsupportedSignatureTypeHintError
from bilt.symbolic._dimensions import Dim, dim
from bilt.symbolic._domain_handling import canonicalised, union
from bilt.symbolic._domains import CanonicalisedDomain, Domain


@dataclasses.dataclass(frozen=True)
class TransformDomainAnalysis:
    parameters: dict[str, CanonicalisedDomain]
    return_domain: CanonicalisedDomain
    union_domain: CanonicalisedDomain


def _extract_domains_from_transform_signature(func_def: ast.FunctionDef) -> dict[str, Domain]:
    res: dict[str, Domain] = {}

    for arg in [*func_def.args.args, ast.arg("return", annotation=func_def.returns)]:
        name = arg.arg
        annotation = arg.annotation

        match annotation:
            case ast.Subscript():
                value = typing.cast("ast.Name", annotation.value).id
                slice_expr = annotation.slice
                dims: list[Dim] = []

                if value != "Tensor":
                    message = f"Expecting `float` or `Tensor` in signature, got `{value}`."
                    raise UnsupportedSignatureTypeHintError(message)

                def add_dim_from_ast_constant(constant: ast.Constant) -> None:
                    if constant.value is None:
                        # Don't add anything to dims (None means scalar, which has no dim)
                        return

                    if not isinstance(constant.value, str):
                        message = f"Expected string dimensions or `None` inside the `Tensor[...]` type hint, but got type `{type(constant.value)}`."
                        raise ValueError(message)

                    dims.append(dim(constant.value))

                match slice_expr:
                    case ast.Tuple():
                        for elt in slice_expr.elts:
                            if not isinstance(elt, ast.Constant):
                                message = f"Expected `ast.Constant` string dimension inside `Tensor[...]`, but instead got type `{type(elt)}`. Make sure you're not passing a runtime object (such as a `Dim` object) directly."
                                raise ValueError(message)

                            add_dim_from_ast_constant(elt)
                    case ast.Constant():
                        add_dim_from_ast_constant(slice_expr)
                    case _:
                        message = f"Expected `ast.Tuple`/`ast.Constant` string slice expression inside `Tensor[...]` type hint, but got type `{type(slice_expr)}`."
                        raise ValueError(message)

                res[name] = Domain(tuple(dims))
            case ast.Name():
                if annotation.id != "float":
                    message = f"Expecting `float` or `Tensor` in signature, got `{annotation.id}`."
                    raise UnsupportedSignatureTypeHintError(message)

                res[name] = Domain.scalar()
            case _:
                raise NotImplementedError

    return res


def analyse_domains_in_transform_signature(func_def: ast.FunctionDef) -> TransformDomainAnalysis:
    domains = _extract_domains_from_transform_signature(func_def)
    domains = {name: canonicalised(domain) for name, domain in domains.items()}
    union_domain = canonicalised(union(*[domain.domain for domain in domains.values()]))

    return_domain = domains.pop("return")
    parameters = domains

    return TransformDomainAnalysis(
        parameters,
        return_domain,
        union_domain,
    )

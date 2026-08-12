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
                    message = f"Expecting `float` or `Tensor` in signature, got `f{value}`."
                    raise UnsupportedSignatureTypeHintError(message)

                match slice_expr:
                    case ast.Tuple():
                        dims.extend([dim(elem.id) for elem in typing.cast("list[ast.Name]", slice_expr.elts)])
                    case ast.Constant():
                        if slice_expr.value is not None:
                            message = f"Expected `Tensor[None]` but given `Tensor[f{slice_expr.value}]"
                            raise UnsupportedSignatureTypeHintError(message)
                        # dims will be left empty (scalar domain)
                    case ast.Name():
                        dims.append(dim(slice_expr.id))
                    case _:
                        message = f"Expected Tuple/Constant/Name slice expression but got {slice_expr}."
                        raise ValueError(message)

                res[name] = Domain(tuple(dims))
            case ast.Name():
                if annotation.id != "float":
                    message = f"Expecting `float` or `Tensor` in signature, got `f{annotation.id}`."
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

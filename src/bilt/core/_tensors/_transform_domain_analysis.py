import ast
import typing

from bilt.symbolic._dimensions import Dim, dim
from bilt.symbolic._domain_handling import canonicalised, union
from bilt.symbolic._domains import CanonicalisedDomain, Domain


import dataclasses


@dataclasses.dataclass(frozen=True)
class TransformDomainAnalysis:
    parameters: dict[str, CanonicalisedDomain]
    return_domain: CanonicalisedDomain
    union_domain: CanonicalisedDomain


def _extract_domains_from_transform_signature(func_def: ast.FunctionDef) -> dict[str, Domain]:
    res: dict[str, Domain] = {}

    # TODO: What about posonly and kwonly args
    for arg in func_def.args.args + [ast.arg("return", annotation=func_def.returns)]:
        name = arg.arg
        annotation = arg.annotation

        match annotation:
            case ast.Subscript():
                value = typing.cast(ast.Name, annotation.value).id
                slice = annotation.slice
                dims: list[Dim] = []

                assert value == "Tensor"

                match slice:
                    case ast.Tuple():
                        for elem in typing.cast(list[ast.Name], slice.elts):
                            dims.append(dim(elem.id))
                    case ast.Constant():
                        assert slice.value is None
                        # dims will be left empty (scalar domain)
                    case ast.Name():
                        dims.append(dim(slice.id))
                    case _:
                        raise RuntimeError()

                res[name] = Domain(tuple(dims))
            case ast.Name():
                assert annotation.id == "float"
                res[name] = Domain.scalar()
            case _:
                raise NotImplementedError()

    return res


def analyse_domains_in_transform_signature(func_def: ast.FunctionDef) -> TransformDomainAnalysis:
    domains = _extract_domains_from_transform_signature(func_def)
    domains = {name: canonicalised(domain) for name, domain in domains.items()}
    union_domain = canonicalised(union(*[domain.domain for domain in domains.values()]))

    return_domain = domains.pop("return")
    parameters = {
        name: domain
        for name, domain in domains.items()
    }

    return TransformDomainAnalysis(
        parameters,
        return_domain,
        union_domain,
    )
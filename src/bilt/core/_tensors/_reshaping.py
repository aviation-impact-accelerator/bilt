from bilt.symbolic._domains import CanonicalisedDomain


type ReshapeSlices = tuple[slice | None, ...]

SLICE_COLON = slice(None)
SLICE_NEWAXIS = None


def derive_reshape_slices(domain: CanonicalisedDomain, target_domain: CanonicalisedDomain) -> ReshapeSlices:
    reshape_slices: list[slice | None] = []

    domain_index = 0

    def any_domain_dims_left():
        return domain_index < len(domain.domain.dims)

    for target_domain_index in range(len(target_domain.domain.dims)):
        target_domain_dim = target_domain.domain.dims[target_domain_index]

        if any_domain_dims_left() and domain.domain.dims[domain_index] == target_domain_dim:
            reshape_slices.append(SLICE_COLON)
            domain_index += 1
        else:
            reshape_slices.append(SLICE_NEWAXIS)

    if any_domain_dims_left():
        message = "Went all the way through `target_domain` and still have `domain` dims left - " \
        "`domain` and `target_domain` not in the same order? Shouldn't be possible now we're taking " \
        "in `CanonicalisedDomain`s."

        raise RuntimeError(message)

    return tuple(reshape_slices)


def is_scalar_reshape(reshape_slices: ReshapeSlices) -> bool:
    for slice in reshape_slices:
        if slice != SLICE_NEWAXIS:
            return False

    return True


def is_unchanged_reshape(reshape_slices: ReshapeSlices) -> bool:
    for slice in reshape_slices:
        if slice != SLICE_COLON:
            return False

    return True
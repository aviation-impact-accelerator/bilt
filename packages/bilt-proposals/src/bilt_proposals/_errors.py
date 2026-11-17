"""The error every proposed API raises when used."""

import typing


class ProposalNotImplementedError(NotImplementedError):
    """Raised when code uses a proposed Bilt API that has not been implemented."""

    def __init__(self, feature: str) -> None:
        super().__init__(f"{feature} is a proposed Bilt API (bilt_proposals) and is not implemented.")
        self.feature = feature


def not_implemented(feature: str) -> typing.NoReturn:
    """Raise `ProposalNotImplementedError` for `feature`."""
    raise ProposalNotImplementedError(feature)

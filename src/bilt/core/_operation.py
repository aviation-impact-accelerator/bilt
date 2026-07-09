import typing


class BiltOperation(typing.Protocol):
    __bilt_name__: str
    __bilt_parameters__: tuple[str, ...]

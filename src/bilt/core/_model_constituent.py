import typing

if typing.TYPE_CHECKING:
    from bilt.core._port_declaration import PortDeclaration


class BiltModelConstituent(typing.Protocol):
    __bilt_name__: str
    __bilt_parameters__: tuple[str, ...]
    __bilt_port_declarations__: dict[str, PortDeclaration]

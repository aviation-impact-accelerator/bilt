class BiltBackendError(Exception):
    pass


class UnsupportedNodeError(BiltBackendError):
    pass


class CompilationError(BiltBackendError):
    pass

class TensorDomainError(ValueError):
    """Raised when concrete tensor values do not match their declared domain."""

class UnsupportedSignatureTypeHintError(ValueError):
    """Raised when a type hint given in a transform/equation signature is not supported."""

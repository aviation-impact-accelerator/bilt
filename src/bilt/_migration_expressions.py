"""Generate Python expressions while preserving their intended AST structure."""

import ast

_BINARY_OPERATORS = frozenset({"+", "-", "*", "@", "/", "//", "%", "**", "<<", ">>", "&", "^", "|"})


def _parenthesize(source: bytes) -> bytes:
    return b"(" + source + b")"


def _tree(source: bytes, *, expression: bool = False) -> str | None:
    """Return a comparable AST or None for a syntactically invalid candidate."""
    try:
        tree = ast.parse(source.decode(), mode="eval" if expression else "exec")
    except SyntaxError:
        return None
    return ast.dump(tree)


def binary_expression(left: bytes, operator: str, right: bytes) -> bytes:
    """Compose operands without changing precedence, grouping or evaluation order.

    Operands are already rendered source, including any nested migration edits.
    Compare candidates with the fully parenthesized reference AST rather than
    assuming associative operators can be regrouped.
    """
    if operator not in _BINARY_OPERATORS:
        msg = f"Unsupported binary operator: {operator!r}"
        raise ValueError(msg)
    separator = f" {operator} ".encode()
    wrapped_left = _parenthesize(left)
    wrapped_right = _parenthesize(right)
    reference = wrapped_left + separator + wrapped_right
    expected = _tree(_parenthesize(reference), expression=True)
    if expected is None:
        msg = "Cannot compose binary expression from invalid operands"
        raise ValueError(msg)
    for candidate in (left + separator + right, wrapped_left + separator + right, left + separator + wrapped_right):
        if _tree(_parenthesize(candidate), expression=True) == expected:
            return candidate
    return reference


def contextual_expression(source: bytes, start: int, end: int, replacement: bytes) -> bytes:
    """Omit surrounding parentheses only if the entire source parses identically.

    Checking the enclosing module handles operator precedence, function arguments,
    attribute access, comprehensions, f-strings and other grammar contexts without
    having to duplicate the Python grammar in the migration framework.
    """
    wrapped = _parenthesize(replacement)
    expected = _tree(source[:start] + wrapped + source[end:])
    if expected is None:
        msg = "Cannot insert replacement expression in its original context"
        raise ValueError(msg)
    if _tree(source[:start] + replacement + source[end:]) == expected:
        return replacement
    return wrapped

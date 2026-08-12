import typing

if typing.TYPE_CHECKING:
    import collections.abc

    from bilt.symbolic._tree import Tree

__all__ = [
    "contains",
    "subtrees",
]


def _walk_subtrees(root: Tree) -> collections.abc.Iterator[Tree]:
    """Yield each distinct subtree of root once."""
    seen: set[Tree] = set()
    stack = [root]

    while stack:
        node = stack.pop()

        if node in seen:
            continue

        seen.add(node)
        yield node
        stack.extend(node.args)


def subtrees(root: Tree) -> set[Tree]:
    """Return every distinct subtree of root."""
    return set(_walk_subtrees(root))


def contains(root: Tree, target: Tree) -> bool:
    """Return True if target occurs anywhere inside root."""
    return any(node is target for node in _walk_subtrees(root))


def count_occurrences(root: Tree, target: Tree) -> int:
    """Count how many times target occurs inside root.

    Repeated references are counted as repeated occurrences.

    This matters for hash-consed trees, where x + x may contain the same
    target node twice.
    """
    count = 0
    stack = [root]

    while stack:
        node = stack.pop()

        if node is target:
            count += 1

        stack.extend(node.args)

    return count

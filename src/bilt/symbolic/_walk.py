import typing

if typing.TYPE_CHECKING:
    from bilt.symbolic._tree import Tree


__all__ = [
    "contains",
    "subtrees",
]


def subtrees(root: Tree) -> set[Tree]:
    """Return every distinct subtree of root, in a single DAG walk.

    Trees are hash-consed, so testing `target in subtrees(root)` is equivalent to
    `contains(root, target)`. Prefer this when several targets are tested against
    one root: it pays for a single traversal rather than one per target.
    """
    seen: set[Tree] = set()
    stack: list[Tree] = [root]

    while stack:
        node = stack.pop()

        if node in seen:
            continue

        seen.add(node)
        stack.extend(node.args)

    return seen


def contains(root: Tree, target: Tree) -> bool:
    """Return True if target occurs anywhere inside root.

    Uses hash-consing by treating the expression as a DAG and avoiding
    repeated visits to shared subtrees.
    """
    seen: set[Tree] = set()
    stack: list[Tree] = [root]

    while stack:
        node = stack.pop()

        if node is target:
            return True

        if node in seen:
            continue

        seen.add(node)
        stack.extend(node.args)

    return False


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

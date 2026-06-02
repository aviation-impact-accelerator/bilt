import typing

if typing.TYPE_CHECKING:
    from bilt.symbolic._tree import Tree


__all__ = [
    "contains",
]

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

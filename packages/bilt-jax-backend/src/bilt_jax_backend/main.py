"""File for playing around with bytecode for JAX backend."""

import dis

__all__ = ()


def c(a: int) -> int:
    return a + 1


dis.dis(c.__code__)

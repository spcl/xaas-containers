from typing import TypeVar, Callable

T = TypeVar('T')
K = TypeVar('K')
V = TypeVar('V')

def union_distinct(a: dict[K, V], b: dict[K, V]) -> dict[K, V]:
    """
    Returns a new dict which is the union of the two given dicts.
    If any key is present in both dicts, a RuntimeError is raised.
    """

    if not a.keys().isdisjoint(b.keys()):
        raise RuntimeError(f"duplicate keys: {a.keys() & b.keys()}")

    return a | b


def union_merge(a: dict[K, V], b: dict[K, V], merge_reduce: Callable[[V, V], V]) -> dict[K, V]:
    """
    Returns a new dict which is the union of the two given dicts.
    If any key is present in both dicts, the given function is used to merge the corresponding values.
    """

    result = a | b
    for k in a.keys() & b.keys():
        result[k] = merge_reduce(a[k], b[k])
    return result

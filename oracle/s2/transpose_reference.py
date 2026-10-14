def valid_axes(rank: int, a: int, b: int) -> bool:
    limit = 1 if rank == 0 else rank
    return 0 <= a < limit and 0 <= b < limit


def requested_axes(rank: int, a: int, b: int) -> list[int]:
    if not valid_axes(rank, a, b):
        raise ValueError("Transpose axis out of bounds")
    if rank == 0:
        return []
    axes = list(range(rank))
    axes[a], axes[b] = axes[b], axes[a]
    return axes

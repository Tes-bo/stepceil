"""Half-open nanosecond intervals. Touching endpoints do not overlap."""


def merge(spans):
    """Union of [start, end) spans. Touching spans become one span."""
    ordered = sorted((int(start), int(end)) for start, end in spans if end > start)
    if not ordered:
        return []
    merged = [ordered[0]]
    for start, end in ordered[1:]:
        previous_start, previous_end = merged[-1]
        if start <= previous_end:
            merged[-1] = (previous_start, max(previous_end, end))
        else:
            merged.append((start, end))
    return merged


def length(spans):
    return sum(end - start for start, end in spans)


def complement(spans, lo, hi):
    """Holes of a merged, in-window span list inside [lo, hi)."""
    holes = []
    cursor = lo
    for start, end in spans:
        if start > cursor:
            holes.append((cursor, start))
        cursor = end
    if cursor < hi:
        holes.append((cursor, hi))
    return holes

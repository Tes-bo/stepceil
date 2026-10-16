"""Account one NVTX step.

Device time blocks the step only when the clipped device intervals do not
overlap. The ceiling of the hottest kernel is the sum of its clipped
launches. Overlap withholds that ceiling. A synchronization call counts
only on idle time; time already covered by a kernel, copy, or memset stays
on that operation.
"""

from dataclasses import replace

from stepceil import REWRITE_DENOMINATOR, REWRITE_NUMERATOR
from stepceil.errors import StepceilError
from stepceil.intervals import complement, length, merge

ASSUMPTION_EXCLUSIVE = "single stream, no overlapping device work"
ASSUMPTION_MULTI = "overlapping device work on more than one stream"
ASSUMPTION_SAME = "device activities overlap on one stream"
GPU_IDLE_NAME = "gpu idle"

KIND_RANK = {
    "kernel": 0,
    "memcpy": 1,
    "memset": 2,
    "sync": 3,
    "gap": 4,
}


def account(window):
    if window.end_ns <= window.start_ns:
        raise StepceilError(
            f"NVTX range {window.name!r} ends at or before it starts"
        )
    clipped = []
    for op in window.device_ops:
        piece = _clip_op(op, window.start_ns, window.end_ns)
        if piece is not None:
            clipped.append(piece)
    _require_identity(
        clipped, window.name, window.syncs, window.start_ns, window.end_ns,
    )

    step_time = window.end_ns - window.start_ns
    busy_spans = merge((op.start_ns, op.end_ns) for op in clipped)
    busy = length(busy_spans)
    idle = step_time - busy
    if busy + idle != step_time or busy < 0 or idle < 0:
        raise RuntimeError("busy and idle do not add up to the step")

    overlap_kind, overlap_pair = _scan_overlap(clipped)
    if overlap_kind == "multi":
        return _refused(window, busy, idle, ASSUMPTION_MULTI, overlap_pair)
    if overlap_kind == "same":
        return _refused(window, busy, idle, ASSUMPTION_SAME, overlap_pair)

    holes = complement(busy_spans, window.start_ns, window.end_ns)
    candidates = [_op_candidate(op) for op in clipped]
    candidates.extend(
        _candidate("sync", sync.name, start, end)
        for start, end, sync in _sync_pieces(holes, window.syncs)
    )
    candidates.extend(
        _candidate("gap", GPU_IDLE_NAME, start, end)
        for start, end in _gap_pieces(holes, window.syncs)
    )
    longest = _longest(candidates)
    hottest, ceiling = _hottest([op for op in clipped if op.kind == "kernel"])
    allow = _allows_rewrite(ceiling, step_time)
    return _ledger(
        window,
        busy,
        idle,
        ASSUMPTION_EXCLUSIVE,
        longest,
        hottest,
        ceiling,
        allow,
        None,
        _decision(longest, hottest, allow, None),
        None,
    )


def overlap_class(ops):
    """Return None, 'multi', or 'same' for clipped device operations."""
    kind, _pair = _scan_overlap(ops)
    return kind


def _order_key(op):
    """Total order for a device op.

    Start, end, and stream are not enough: an export can return two
    launches with those three fields equal in either row order. Context
    and name finish the key so the witness does not follow that order.
    A missing context sorts after a real one.
    """
    context = op.context_id
    return (
        op.start_ns,
        op.end_ns,
        op.stream_id,
        context is None,
        0 if context is None else context,
        op.kind,
        op.name,
    )


def _scan_overlap(ops):
    """Classify overlap, and keep the longest pair of that class.

    A cross-stream pair wins over a same-stream pair, because that is
    the class the ledger reports. Within the class, the longest overlap
    is the witness. Ties keep the earlier overlap.
    """
    ordered = sorted(ops, key=_order_key)
    if not ordered:
        return None, None
    current = ordered[0]
    best_multi = None
    best_same = None
    for nxt in ordered[1:]:
        if nxt.start_ns < current.end_ns:
            ranked = _ranked_pair(current, nxt)
            if _stream_key(nxt) != _stream_key(current):
                if best_multi is None or ranked[0] > best_multi[0]:
                    best_multi = ranked
            else:
                if best_same is None or ranked[0] > best_same[0]:
                    best_same = ranked
            if nxt.end_ns > current.end_ns:
                current = nxt
        else:
            current = nxt
    if best_multi is not None:
        return "multi", best_multi[1]
    if best_same is not None:
        return "same", best_same[1]
    return None, None


def _ranked_pair(left, right):
    """Return a sort key and the pair, earlier start first.

    Larger overlap wins. An equal overlap keeps the earlier one.
    """
    if _order_key(right) < _order_key(left):
        left, right = right, left
    overlap_start = max(left.start_ns, right.start_ns)
    overlap_ns = min(left.end_ns, right.end_ns) - overlap_start
    rank = (overlap_ns, -overlap_start, -left.start_ns, left.name, right.name)
    return rank, (left, right)


def _clip_op(op, lo, hi):
    start = max(op.start_ns, lo)
    end = min(op.end_ns, hi)
    if end <= start:
        return None
    # The bytes column counts the whole transfer. A clipped interval is only
    # the piece inside the step, so that size would not match the duration.
    byte_count = op.byte_count if start == op.start_ns and end == op.end_ns else None
    return replace(op, start_ns=start, end_ns=end, byte_count=byte_count)


def _require_identity(ops, range_name, syncs=(), lo=None, hi=None):
    if any(op.device_id is None for op in ops):
        raise StepceilError("a device operation has no device id")
    if any(op.stream_id is None for op in ops):
        raise StepceilError(
            "a device operation has no stream id; overlap cannot be judged"
        )
    devices = set()
    for op in ops:
        devices.add(op.device_id)
        devices.update(op.peer_device_ids)
    devices = sorted(devices)
    if len(devices) > 1:
        ids = ", ".join(str(device) for device in devices)
        raise StepceilError(
            f"more than one GPU in NVTX range {range_name!r} "
            f"(device ids {ids}); this version does not account that"
        )
    # Stream ids are per process. Two processes in one export reuse them,
    # so the other process is not this step's stream and not its ceiling.
    processes = {op.process_id for op in ops if op.process_id is not None}
    if lo is not None:
        for sync in syncs:
            if sync.process_id is None:
                continue
            if sync.end_ns > lo and sync.start_ns < hi:
                processes.add(sync.process_id)
    processes = sorted(processes)
    if len(processes) > 1:
        ids = ", ".join(str(process) for process in processes)
        raise StepceilError(
            f"more than one process in NVTX range {range_name!r} "
            f"(process ids {ids}); this version does not account that"
        )


def _stream_key(op):
    return (op.context_id, op.stream_id)


def _sync_pieces(holes, syncs):
    """Idle covered by each synchronization, one contiguous piece at a time.

    A shorter call that overlaps a longer one does not take that idle away.
    The ledger reports the longest piece, so both calls keep their own
    coverage. A kernel, copy, or memset between two pieces still splits
    them: those pieces are not one blocker.
    """
    pieces = []
    for sync in syncs:
        spans = []
        for hole_start, hole_end in holes:
            start = max(sync.start_ns, hole_start)
            end = min(sync.end_ns, hole_end)
            if end > start:
                spans.append((start, end))
        for start, end in merge(spans):
            pieces.append((start, end, sync))
    return pieces


def _gap_pieces(holes, syncs):
    gaps = []
    for hole_start, hole_end in holes:
        covered = []
        for sync in syncs:
            start = max(sync.start_ns, hole_start)
            end = min(sync.end_ns, hole_end)
            if end > start:
                covered.append((start, end))
        gaps.extend(complement(merge(covered), hole_start, hole_end))
    return merge(gaps)


def _op_candidate(op):
    return _candidate(
        op.kind, op.name, op.start_ns, op.end_ns, op.byte_count
    )


def _candidate(kind, name, start, end, byte_count=None):
    return {
        "kind": kind,
        "name": name,
        "start_ns": start,
        "end_ns": end,
        "blocking_ns": end - start,
        "source": None,
        "bytes": byte_count,
    }


def _longest(candidates):
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda item: (
            -item["blocking_ns"],
            item["start_ns"],
            KIND_RANK[item["kind"]],
            item["name"],
        ),
    )


def _hottest(kernels):
    if not kernels:
        return None, 0
    groups = {}
    for op in kernels:
        duration = op.end_ns - op.start_ns
        ceiling, count, first = groups.get(op.name, (0, 0, op.start_ns))
        groups[op.name] = (ceiling + duration, count + 1, min(first, op.start_ns))
    name = min(groups, key=lambda item: (-groups[item][0], groups[item][2], item))
    ceiling, count, _first = groups[name]
    return {
        "name": name,
        "ceiling_ns": ceiling,
        "instances": count,
        "source": None,
    }, ceiling


def _allows_rewrite(ceiling, step_time):
    return ceiling * REWRITE_DENOMINATOR >= step_time * REWRITE_NUMERATOR


def _decision(longest, hottest, allow, refused):
    """The action the agent must take. It is not allow_rewrite by another name.

    allow_rewrite can be true while a synchronize or a gap is still the
    longest blocker. Rewrite is only the case where that flag is true and
    the hottest kernel is that blocker.
    """
    if refused is not None or longest is None:
        return "do not rewrite a kernel from this report"
    if (
        allow
        and hottest is not None
        and longest["kind"] == "kernel"
        and longest["name"] == hottest["name"]
    ):
        return f"rewrite {hottest['name']}"
    return f"change {longest['name']} first"


def _refused(window, busy, idle, reason, pair):
    return _ledger(
        window, busy, idle, reason, None, None, None, False, reason,
        _decision(None, None, False, reason),
        _overlap_payload(pair),
    )


def _overlap_payload(pair):
    if pair is None:
        return None
    first, second = pair
    overlap_ns = (
        min(first.end_ns, second.end_ns)
        - max(first.start_ns, second.start_ns)
    )
    return {
        "overlap_ns": overlap_ns,
        "first": _overlap_op(first),
        "second": _overlap_op(second),
    }


def _overlap_op(op):
    return {
        "kind": op.kind,
        "name": op.name,
        "start_ns": op.start_ns,
        "end_ns": op.end_ns,
        "stream_id": op.stream_id,
        "context_id": op.context_id,
        "bytes": op.byte_count,
    }


def _ledger(
    window, busy, idle, assumption, longest, hottest, ceiling, allow, refused,
    decision, overlap,
):
    return {
        "range": window.name,
        "step_start_ns": window.start_ns,
        "step_end_ns": window.end_ns,
        "step_time_ns": window.end_ns - window.start_ns,
        "gpu_busy_ns": busy,
        "gpu_idle_ns": idle,
        "assumption": assumption,
        "longest_blocker": longest,
        "hottest_kernel": hottest,
        "ceiling_ns": ceiling,
        "allow_rewrite": allow,
        "ceiling_refused": refused,
        "overlap": overlap,
        "rewrite_threshold_numerator": REWRITE_NUMERATOR,
        "rewrite_threshold_denominator": REWRITE_DENOMINATOR,
        "decision": decision,
    }

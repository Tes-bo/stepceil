"""Read one NVTX step from an nsys sqlite export.

The export is the file `nsys export --type sqlite` writes. This module does
not open a `.nsys-rep`.

Only rows that meet the step are read from the CUPTI tables, and only the
StringIds those rows name are resolved. A real export holds millions of
runtime rows and strings for the whole capture.
"""

import sqlite3
from contextlib import closing
from pathlib import Path
from typing import NamedTuple

from stepceil.errors import StepceilError
from stepceil.model import DeviceOp, NvtxOp, RuntimeOp, StepWindow, SyncOp

NVTX_PUSH_POP = 59
NVTX_START_END = 60

COPY_KIND = {
    0: "unknown",
    1: "HtoD",
    2: "DtoH",
    3: "HtoH",
    4: "DtoD",
    5: "DtoA",
    6: "AtoD",
    7: "AtoH",
    8: "HtoA",
    9: "AtoA",
    10: "AtoP",
    11: "PtoA",
    12: "PtoP",
}

SYNC_TYPE = {
    0: "unknown synchronize",
    1: "event synchronize",
    2: "stream wait event",
    3: "stream synchronize",
    4: "context synchronize",
}

KERNEL_TABLE = "CUPTI_ACTIVITY_KIND_KERNEL"
MEMCPY_TABLE = "CUPTI_ACTIVITY_KIND_MEMCPY"
MEMSET_TABLE = "CUPTI_ACTIVITY_KIND_MEMSET"
SYNC_TABLE = "CUPTI_ACTIVITY_KIND_SYNCHRONIZATION"
RUNTIME_TABLE = "CUPTI_ACTIVITY_KIND_RUNTIME"

DEVICE_COLUMNS = ("start", "end", "deviceId", "streamId")
KERNEL_NAME_COLUMNS = ("shortName", "demangledName", "mangledName")
SYMBOL_COLUMNS = ("mangledName", "demangledName")
UNRESOLVED = "unresolved"
# Nsight joins a runtime row to a device row with
# globalPid == (globalTid & PROCESS_TID_MASK). The low 24 bits are the thread.
PROCESS_TID_MASK = 0xFFFFFFFFFF000000
_IN_CHUNK = 500


class _Event(NamedTuple):
    name: str
    start: int | None
    end: int | None
    process_id: int | None


class _Call(NamedTuple):
    name: str
    start: int
    end: int
    process_id: int | None
    correlation_id: int | None


def load_path(path, range_name, occurrence=None):
    """The step window of one NVTX range.

    Without occurrence the name must occur once. With occurrence N, the
    step is the Nth closed occurrence of that name, ordered by start time,
    counting from 1. That is the `closed` count of list_ranges.
    """
    if range_name == "":
        raise StepceilError("empty range name")
    if occurrence is not None and (
        isinstance(occurrence, bool)
        or not isinstance(occurrence, int)
        or occurrence < 1
    ):
        raise StepceilError("occurrence must be an integer of at least 1")
    with closing(_open(path)) as connection:
        return _load(connection, range_name, occurrence)


def list_ranges(path):
    """NVTX range names in this export.

    The accountant accepts a name only when it occurs once and is closed.
    shortest_ns and longest_ns are the closed occurrences. They do not
    choose a window.
    """
    with closing(_open(path)) as connection:
        events = _nvtx_events(connection, _Strings(connection))
    counts = {}
    durations = {}
    for event in events:
        counts[event.name] = counts.get(event.name, 0) + 1
        closed = durations.setdefault(event.name, [])
        if _closed(event.start, event.end):
            closed.append(event.end - event.start)
    ranges = []
    for name in sorted(counts):
        closed = durations[name]
        ranges.append({
            "name": name,
            "count": counts[name],
            "closed": len(closed),
            "shortest_ns": min(closed) if closed else None,
            "longest_ns": max(closed) if closed else None,
        })
    return {"ranges": ranges}


def _open(path):
    path = Path(path)
    if not path.is_file():
        raise StepceilError(f"no such file: {path}")
    uri = path.resolve().as_uri() + "?mode=ro"
    try:
        connection = sqlite3.connect(uri, uri=True)
    except sqlite3.Error as exc:
        raise StepceilError(f"not an nsys sqlite export: {path}") from exc
    connection.row_factory = sqlite3.Row
    try:
        connection.execute("SELECT name FROM sqlite_master LIMIT 1")
    except sqlite3.DatabaseError as exc:
        connection.close()
        raise StepceilError(f"not an nsys sqlite export: {path}") from exc
    return connection


def _load(connection, range_name, occurrence):
    strings = _Strings(connection)
    events = _nvtx_events(connection, strings)
    matches = [event for event in events if event.name == range_name]
    if occurrence is None:
        step = _only_match(range_name, matches)
        occurrence = 1
    else:
        step = _nth_closed(range_name, matches, occurrence)
    bounds = (step.start, step.end)
    calls = _runtime_calls(connection, strings, bounds)
    return StepWindow(
        name=range_name,
        start_ns=step.start,
        end_ns=step.end,
        device_ops=(
            *_kernels(connection, strings, bounds),
            *_memcpys(connection, bounds),
            *_memsets(connection, bounds),
        ),
        syncs=tuple(_syncs(connection, bounds, calls)),
        runtime_ops=tuple(
            RuntimeOp(call.name, call.start, call.end, call.process_id)
            for call in calls
        ),
        nvtx_ops=tuple(_nvtx_ops(events, step)),
        process_id=step.process_id,
        occurrence=occurrence,
    )


def _only_match(range_name, matches):
    if len(matches) != 1:
        hint = "; choose one with --occurrence N" if len(matches) > 1 else ""
        raise StepceilError(
            f"expected one NVTX range named {range_name!r}, "
            f"found {len(matches)}{hint}"
        )
    step = matches[0]
    if step.start is None:
        raise StepceilError(f"NVTX range {range_name!r} has no start timestamp")
    if step.end is None:
        raise StepceilError(f"NVTX range {range_name!r} has no end timestamp")
    if step.end <= step.start:
        raise StepceilError(
            f"NVTX range {range_name!r} ends at or before it starts"
        )
    return step


def _nth_closed(range_name, matches, occurrence):
    """The Nth closed occurrence by start, then end, then process.

    An open or empty occurrence is not counted, so N matches the closed
    column of list_ranges.
    """
    closed = sorted(
        (event for event in matches if _closed(event.start, event.end)),
        key=lambda event: (
            event.start,
            event.end,
            event.process_id is None,
            event.process_id or 0,
        ),
    )
    if occurrence > len(closed):
        raise StepceilError(
            f"NVTX range {range_name!r} has {len(closed)} closed occurrences; "
            f"occurrence {occurrence} does not exist"
        )
    return closed[occurrence - 1]


class _Strings:
    """StringIds, read only for the ids that are asked for.

    A value that is already text is returned as it is. An id that is not
    in the table resolves to None.
    """

    def __init__(self, connection):
        _require_table(connection, "StringIds")
        _require_columns(connection, "StringIds", ("id", "value"))
        self._connection = connection
        self._cache = {}

    def prefetch(self, values):
        wanted = {
            int(value) for value in values
            if value is not None and not isinstance(value, str)
        }
        missing = sorted(wanted - self._cache.keys())
        for index in range(0, len(missing), _IN_CHUNK):
            chunk = missing[index:index + _IN_CHUNK]
            marks = ", ".join("?" * len(chunk))
            query = f"SELECT id, value FROM StringIds WHERE id IN ({marks})"
            for key, text in self._connection.execute(query, chunk):
                self._cache[int(key)] = "" if text is None else str(text)
            for key in chunk:
                self._cache.setdefault(key, None)

    def get(self, value):
        if value is None:
            return None
        if isinstance(value, str):
            return value
        key = int(value)
        if key not in self._cache:
            self.prefetch((key,))
        return self._cache[key]


def _nvtx_events(connection, strings):
    """Every push/pop and start/end range, with its text resolved.

    The text column wins when it is non-empty, otherwise StringIds at
    textId. Marks are not ranges.
    """
    _require_table(connection, "NVTX_EVENTS")
    _require_columns(connection, "NVTX_EVENTS", ("start", "end", "eventType"))
    columns = _columns(connection, "NVTX_EVENTS")
    if "text" not in columns and "textId" not in columns:
        raise StepceilError("NVTX_EVENTS has no text or textId column")
    optional = [name for name in ("text", "textId", "globalTid") if name in columns]
    query = (
        f"SELECT {', '.join(['start', 'end', *optional])} FROM NVTX_EVENTS "
        "WHERE eventType IN (?, ?)"
    )
    rows = connection.execute(query, (NVTX_PUSH_POP, NVTX_START_END)).fetchall()
    strings.prefetch(
        _value(row, "textId") for row in rows if not _value(row, "text")
    )
    events = []
    for row in rows:
        text = _value(row, "text")
        if not text:
            text = strings.get(_value(row, "textId"))
        events.append(_Event(
            name=str(text) if text else "",
            start=_int_or_none(row["start"]),
            end=_int_or_none(row["end"]),
            process_id=_process_of_thread(_value(row, "globalTid")),
        ))
    return events


def _nvtx_ops(events, step):
    """Ranges inside the step, other than ranges with the step's name.

    The step range covers every idle piece, so naming it would hide the
    code that actually ran. Another occurrence of the same name is an
    earlier or later step, not code inside this one. An empty name is
    not a blocker.
    """
    ops = []
    for event in events:
        if not event.name or not _closed(event.start, event.end):
            continue
        if event.end <= step.start or event.start >= step.end:
            continue
        if event.name == step.name:
            continue
        ops.append(NvtxOp(event.name, event.start, event.end, event.process_id))
    return ops


def _closed(start, end):
    return start is not None and end is not None and end > start


def _kernels(connection, strings, bounds):
    if not _has_table(connection, KERNEL_TABLE):
        return []
    columns = _columns(connection, KERNEL_TABLE)
    name_columns = [name for name in KERNEL_NAME_COLUMNS if name in columns]
    if not name_columns:
        raise StepceilError(f"{KERNEL_TABLE} is missing a kernel name column")
    rows = _device_rows(connection, KERNEL_TABLE, columns, bounds, name_columns)
    strings.prefetch(row[column] for row in rows for column in name_columns)
    return [
        _device_op(
            "kernel", KERNEL_TABLE, _kernel_name(row, name_columns, strings), row,
            symbol=_kernel_symbol(row, name_columns, strings),
        )
        for row in rows
    ]


def _kernel_name(row, name_columns, strings):
    """shortName, then demangledName, then mangledName, as the row has them."""
    for column in name_columns:
        text = strings.get(row[column])
        if text:
            return text
    return UNRESOLVED


def _kernel_symbol(row, name_columns, strings):
    """The full symbol of one launch, or None when the export has none.

    shortName drops template arguments, so two instances can share it.
    mangledName, then demangledName, tells them apart.
    """
    for column in SYMBOL_COLUMNS:
        if column in name_columns:
            text = strings.get(row[column])
            if text:
                return text
    return None


def _memcpys(connection, bounds):
    if not _has_table(connection, MEMCPY_TABLE):
        return []
    columns = _columns(connection, MEMCPY_TABLE)
    extra = [
        name for name in ("copyKind", "srcDeviceId", "dstDeviceId", "bytes")
        if name in columns
    ]
    labels = _enum_labels(connection, "ENUM_CUDA_MEMCPY_OPER")
    rows = _device_rows(connection, MEMCPY_TABLE, columns, bounds, extra)
    ops = []
    for row in rows:
        if "copyKind" in extra:
            name = _memcpy_name(row["copyKind"], labels)
        else:
            name = "memcpy"
        ops.append(_device_op(
            "memcpy", MEMCPY_TABLE, name, row, byte_count=_value(row, "bytes"),
        ))
    return ops


def _memsets(connection, bounds):
    if not _has_table(connection, MEMSET_TABLE):
        return []
    columns = _columns(connection, MEMSET_TABLE)
    extra = ["bytes"] if "bytes" in columns else []
    rows = _device_rows(connection, MEMSET_TABLE, columns, bounds, extra)
    return [
        _device_op(
            "memset", MEMSET_TABLE, "memset", row, byte_count=_value(row, "bytes"),
        )
        for row in rows
    ]


def _device_rows(connection, table, columns, bounds, extra):
    _require_columns(connection, table, DEVICE_COLUMNS)
    optional = [name for name in ("contextId", "globalPid") if name in columns]
    selected = [*DEVICE_COLUMNS, *optional, *extra]
    query = (
        f"SELECT {', '.join(selected)} FROM {table} "
        "WHERE end > ? AND start < ?"
    )
    return connection.execute(query, bounds).fetchall()


def _device_op(kind, table, name, row, byte_count=None, symbol=None):
    device_id = _required_int(row, "deviceId", table)
    return DeviceOp(
        kind=kind,
        name=name,
        start_ns=_required_int(row, "start", table),
        end_ns=_required_int(row, "end", table),
        stream_id=_required_int(row, "streamId", table),
        device_id=device_id,
        context_id=_int_or_none(_value(row, "contextId")),
        peer_device_ids=_peer_device_ids(row, device_id),
        byte_count=_int_or_none(byte_count),
        process_id=_int_or_none(_value(row, "globalPid")),
        symbol=symbol,
    )


def _peer_device_ids(row, device_id):
    """Other GPUs named on a copy.

    Nsight Systems records a peer copy with deviceId of the context that
    issued it. srcDeviceId and dstDeviceId name the GPUs that actually
    take part. A null column means the export did not name one.
    """
    found = []
    for column in ("srcDeviceId", "dstDeviceId"):
        other = _int_or_none(_value(row, column))
        if other is not None and other != device_id and other not in found:
            found.append(other)
    return tuple(found)


def _syncs(connection, bounds, calls):
    if not _has_table(connection, SYNC_TABLE):
        return []
    _require_columns(connection, SYNC_TABLE, ("start", "end", "syncType"))
    columns = _columns(connection, SYNC_TABLE)
    optional = [name for name in ("correlationId", "globalPid") if name in columns]
    query = (
        f"SELECT {', '.join(['start', 'end', 'syncType', *optional])} "
        f"FROM {SYNC_TABLE} WHERE end > ? AND start < ?"
    )
    rows = connection.execute(query, bounds).fetchall()
    if not rows:
        return []
    labels = _enum_labels(connection, "ENUM_CUPTI_SYNC_TYPE")
    by_correlation = {}
    for call in calls:
        if call.correlation_id is not None:
            by_correlation.setdefault(call.correlation_id, []).append(call)
    syncs = []
    for row in rows:
        start = _required_int(row, "start", SYNC_TABLE)
        end = _required_int(row, "end", SYNC_TABLE)
        if end <= start:
            continue
        sync_type = _required_int(row, "syncType", SYNC_TABLE)
        process_id = _int_or_none(_value(row, "globalPid"))
        name = _covering_sync_name(
            by_correlation.get(_int_or_none(_value(row, "correlationId")), ()),
            start, end, process_id,
        )
        if name is None:
            name = _sync_name(sync_type, labels)
        syncs.append(SyncOp(
            name=name, start_ns=start, end_ns=end, process_id=process_id,
        ))
    return syncs


def _runtime_calls(connection, strings, bounds):
    """Host API calls that meet the step.

    These are not device work. They can name the idle they cover, which is
    how a cudaMallocHost in the middle of a step stops being an unnamed gap.
    They also name a synchronization record they cover. A call that covers
    a record inside the step meets the step, so no other rows are read.
    """
    if not _has_table(connection, RUNTIME_TABLE):
        return []
    columns = _columns(connection, RUNTIME_TABLE)
    if any(name not in columns for name in ("start", "end", "nameId")):
        return []
    optional = [name for name in ("correlationId", "globalTid") if name in columns]
    query = (
        f"SELECT {', '.join(['start', 'end', 'nameId', *optional])} "
        f"FROM {RUNTIME_TABLE} WHERE end > ? AND start < ?"
    )
    rows = connection.execute(query, bounds).fetchall()
    strings.prefetch(row["nameId"] for row in rows)
    calls = []
    for row in rows:
        start = _int_or_none(row["start"])
        end = _int_or_none(row["end"])
        if start is None or end is None or end <= start:
            continue
        name = strings.get(row["nameId"])
        if not name:
            continue
        calls.append(_Call(
            name=name,
            start=start,
            end=end,
            process_id=_process_of_thread(_value(row, "globalTid")),
            correlation_id=_int_or_none(_value(row, "correlationId")),
        ))
    return calls


def _covering_sync_name(calls, start, end, process_id):
    """The runtime API name of a synchronization record, or None.

    CUPTI_ACTIVITY_KIND_SYNCHRONIZATION can label cudaDeviceSynchronize
    as Unknown. The runtime call with the same correlation id that covers
    the record carries the API name. A launch that only overlaps it does
    not. Correlation ids are reused across processes, so a call from
    another process does not count when both sides name a process.
    """
    names = []
    for call in calls:
        if (
            process_id is not None
            and call.process_id is not None
            and call.process_id != process_id
        ):
            continue
        covers = call.start <= start and call.end >= end
        if covers and _is_sync_api(call.name) and call.name not in names:
            names.append(call.name)
    if len(names) == 1:
        return names[0]
    return None


def _is_sync_api(name):
    return "Sync" in name or "Wait" in name


def _enum_labels(connection, table):
    """Names from the export's own enum table.

    Nsight Systems 2022.4 numbers memcpy kinds differently from older CUPTI
    headers. ENUM_CUDA_MEMCPY_OPER in the file is the authority.
    """
    if not _has_table(connection, table):
        return {}
    columns = _columns(connection, table)
    if "id" not in columns or "label" not in columns:
        return {}
    labels = {}
    for row in connection.execute(f"SELECT id, label FROM {table}"):
        if row["id"] is None or not row["label"]:
            continue
        labels[int(row["id"])] = str(row["label"])
    return labels


def _memcpy_name(copy_kind, labels):
    if copy_kind is None:
        return "memcpy unknown"
    kind = int(copy_kind)
    if kind in labels:
        return f"memcpy {labels[kind]}"
    return f"memcpy {COPY_KIND.get(kind, f'kind {kind}')}"


def _sync_name(sync_type, labels):
    if sync_type in labels:
        return labels[sync_type]
    return SYNC_TYPE.get(sync_type, f"synchronize type {sync_type}")


def _process_of_thread(global_tid):
    if global_tid is None:
        return None
    return int(global_tid) & PROCESS_TID_MASK


def _value(row, column):
    """A column that the query selected only when the table has it."""
    return row[column] if column in row.keys() else None


def _int_or_none(value):
    return None if value is None else int(value)


def _has_table(connection, name):
    row = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (name,),
    ).fetchone()
    return row is not None


def _require_table(connection, name):
    if not _has_table(connection, name):
        raise StepceilError(f"sqlite export is missing table {name}")


def _columns(connection, name):
    return {row[1] for row in connection.execute(f"PRAGMA table_info({name})")}


def _require_columns(connection, table, names):
    columns = _columns(connection, table)
    for name in names:
        if name not in columns:
            raise StepceilError(f"{table} is missing column {name}")


def _required_int(row, column, table):
    value = row[column]
    if value is None:
        raise StepceilError(f"{table} has a row with no {column}")
    return int(value)

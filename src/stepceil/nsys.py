"""Read one NVTX step from an nsys sqlite export.

The export is the file `nsys export --type sqlite` writes. This module does
not open a `.nsys-rep`.
"""

import sqlite3
from pathlib import Path

from stepceil.errors import StepceilError
from stepceil.model import DeviceOp, StepWindow, SyncOp

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

KERNEL_NAME_COLUMNS = ("shortName", "demangledName", "mangledName")


def load_path(path, range_name):
    if range_name == "":
        raise StepceilError("empty range name")
    connection = _open(path)
    try:
        return _load(connection, range_name)
    finally:
        connection.close()


def list_ranges(path):
    """NVTX range names in this export.

    The accountant accepts a name only when it occurs once and is closed.
    This list uses those same events. It does not choose a window.
    """
    connection = _open(path)
    try:
        counts = {}
        closed = {}
        for name, start, end in _nvtx_events(connection):
            counts[name] = counts.get(name, 0) + 1
            closed[name] = closed.get(name, 0) + (1 if _closed(start, end) else 0)
        return {
            "ranges": [
                {
                    "name": name,
                    "count": counts[name],
                    "closed": closed[name],
                }
                for name in sorted(counts)
            ]
        }
    finally:
        connection.close()


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


def _load(connection, range_name):
    matches = [
        (start, end)
        for name, start, end in _nvtx_events(connection)
        if name == range_name
    ]
    if len(matches) != 1:
        raise StepceilError(
            f"expected one NVTX range named {range_name!r}, found {len(matches)}"
        )
    start, end = matches[0]
    if end is None:
        raise StepceilError(f"NVTX range {range_name!r} has no end timestamp")
    start_ns = int(start)
    end_ns = int(end)
    if end_ns <= start_ns:
        raise StepceilError(
            f"NVTX range {range_name!r} ends at or before it starts"
        )

    device_ops = []
    strings = _string_ids(connection)
    copy_labels = _enum_labels(connection, "ENUM_CUDA_MEMCPY_OPER")
    sync_labels = _enum_labels(connection, "ENUM_CUPTI_SYNC_TYPE")
    device_ops.extend(_kernels(connection, strings, start_ns, end_ns))
    device_ops.extend(_memcpys(connection, start_ns, end_ns, copy_labels))
    device_ops.extend(_memsets(connection, start_ns, end_ns))
    syncs = _syncs(connection, strings, start_ns, end_ns, sync_labels)
    return StepWindow(
        name=range_name,
        start_ns=start_ns,
        end_ns=end_ns,
        device_ops=tuple(device_ops),
        syncs=tuple(syncs),
    )


def _nvtx_events(connection):
    _require_table(connection, "NVTX_EVENTS")
    _require_columns(connection, "NVTX_EVENTS", ("start", "end", "eventType"))
    columns = _columns(connection, "NVTX_EVENTS")
    has_text = "text" in columns
    has_text_id = "textId" in columns
    if not has_text and not has_text_id:
        raise StepceilError("NVTX_EVENTS has no text or textId column")
    strings = _string_ids(connection)
    selected = ["start", "end", "eventType"]
    if has_text:
        selected.append("text")
    if has_text_id:
        selected.append("textId")
    query = (
        f"SELECT {', '.join(selected)} FROM NVTX_EVENTS "
        "WHERE eventType IN (?, ?)"
    )
    events = []
    for row in connection.execute(query, (NVTX_PUSH_POP, NVTX_START_END)):
        events.append((
            _range_text(row, has_text, has_text_id, strings),
            row["start"],
            row["end"],
        ))
    return events


def _closed(start, end):
    if start is None or end is None:
        return False
    return int(end) > int(start)


def _range_text(row, has_text, has_text_id, strings):
    if has_text:
        text = row["text"]
        if text:
            return str(text)
    if has_text_id and row["textId"] is not None:
        return strings.get(int(row["textId"]), "")
    return ""


def _kernels(connection, strings, start_ns, end_ns):
    table = "CUPTI_ACTIVITY_KIND_KERNEL"
    if not _has_table(connection, table):
        return []
    columns = _columns(connection, table)
    _require_columns(connection, table, ("start", "end", "deviceId", "streamId"))
    name_column = next((name for name in KERNEL_NAME_COLUMNS if name in columns), None)
    if name_column is None:
        raise StepceilError(f"{table} is missing a kernel name column")
    rows = _device_rows(connection, table, columns, start_ns, end_ns, (name_column,))
    ops = []
    for row in rows:
        ops.append(_device_op(
            "kernel",
            _lookup_string(row[name_column], strings),
            row,
            table,
        ))
    return ops


def _memcpys(connection, start_ns, end_ns, copy_labels):
    table = "CUPTI_ACTIVITY_KIND_MEMCPY"
    if not _has_table(connection, table):
        return []
    columns = _columns(connection, table)
    _require_columns(connection, table, ("start", "end", "deviceId", "streamId"))
    extra = tuple(
        name for name in ("copyKind", "srcDeviceId", "dstDeviceId")
        if name in columns
    )
    rows = _device_rows(connection, table, columns, start_ns, end_ns, extra)
    ops = []
    for row in rows:
        if "copyKind" in extra:
            name = _memcpy_name(row["copyKind"], copy_labels)
        else:
            name = "memcpy"
        ops.append(_device_op("memcpy", name, row, table))
    return ops


def _memsets(connection, start_ns, end_ns):
    table = "CUPTI_ACTIVITY_KIND_MEMSET"
    if not _has_table(connection, table):
        return []
    columns = _columns(connection, table)
    _require_columns(connection, table, ("start", "end", "deviceId", "streamId"))
    rows = _device_rows(connection, table, columns, start_ns, end_ns, ())
    return [_device_op("memset", "memset", row, table) for row in rows]


def _syncs(connection, strings, start_ns, end_ns, sync_labels):
    table = "CUPTI_ACTIVITY_KIND_SYNCHRONIZATION"
    if not _has_table(connection, table):
        return []
    _require_columns(connection, table, ("start", "end", "syncType"))
    columns = _columns(connection, table)
    selected = ["start", "end", "syncType"]
    if "correlationId" in columns:
        selected.append("correlationId")
    runtime = _runtime_index(connection, strings)
    query = (
        f"SELECT {', '.join(selected)} FROM {table} "
        "WHERE end > ? AND start < ?"
    )
    syncs = []
    for row in connection.execute(query, (start_ns, end_ns)):
        start = _required_int(row, "start", table)
        end = _required_int(row, "end", table)
        if end <= start:
            continue
        sync_type = _required_int(row, "syncType", table)
        name = _sync_name(sync_type, sync_labels)
        correlation = row["correlationId"] if "correlationId" in row.keys() else None
        if correlation is not None:
            runtime_name = _overlapping_runtime_name(
                runtime, int(correlation), start, end
            )
            if runtime_name:
                name = runtime_name
        syncs.append(SyncOp(name=name, start_ns=start, end_ns=end))
    return syncs


def _device_rows(connection, table, columns, start_ns, end_ns, extra):
    selected = ["start", "end", "deviceId", "streamId"]
    if "contextId" in columns:
        selected.append("contextId")
    selected.extend(extra)
    query = (
        f"SELECT {', '.join(selected)} FROM {table} "
        "WHERE end > ? AND start < ?"
    )
    return list(connection.execute(query, (start_ns, end_ns)))


def _device_op(kind, name, row, table):
    keys = row.keys()
    context_id = None
    if "contextId" in keys and row["contextId"] is not None:
        context_id = int(row["contextId"])
    device_id = _required_int(row, "deviceId", table)
    return DeviceOp(
        kind=kind,
        name=name,
        start_ns=_required_int(row, "start", table),
        end_ns=_required_int(row, "end", table),
        stream_id=_required_int(row, "streamId", table),
        device_id=device_id,
        context_id=context_id,
        peer_device_ids=_peer_device_ids(row, device_id),
    )


def _peer_device_ids(row, device_id):
    """Other GPUs named on a copy.

    Nsight Systems records a peer copy with deviceId of the context that
    issued it. srcDeviceId and dstDeviceId name the GPUs that actually
    take part. A null column means the export did not name one.
    """
    found = []
    keys = row.keys()
    for column in ("srcDeviceId", "dstDeviceId"):
        if column not in keys or row[column] is None:
            continue
        other = int(row[column])
        if other != device_id and other not in found:
            found.append(other)
    return tuple(found)


def _runtime_index(connection, strings):
    """Runtime API calls keyed by correlation id.

    CUPTI_ACTIVITY_KIND_SYNCHRONIZATION labels cudaDeviceSynchronize as
    Unknown in Nsight Systems 2022.4. The runtime row with the same
    correlation id carries the API name.
    """
    table = "CUPTI_ACTIVITY_KIND_RUNTIME"
    if not _has_table(connection, table):
        return {}
    columns = _columns(connection, table)
    needed = ("start", "end", "correlationId", "nameId")
    if any(name not in columns for name in needed):
        return {}
    grouped = {}
    query = f"SELECT {', '.join(needed)} FROM {table}"
    for row in connection.execute(query):
        if row["correlationId"] is None or row["start"] is None or row["end"] is None:
            continue
        name = _lookup_string(row["nameId"], strings)
        if not name or name == "unresolved" or name.startswith("unresolved:"):
            continue
        grouped.setdefault(int(row["correlationId"]), []).append(
            (int(row["start"]), int(row["end"]), name)
        )
    return grouped


def _overlapping_runtime_name(runtime, correlation_id, start, end):
    names = []
    for call_start, call_end, name in runtime.get(correlation_id, ()):
        if call_end > start and call_start < end and name not in names:
            names.append(name)
    if len(names) == 1:
        return names[0]
    return None


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
    label = COPY_KIND.get(kind, f"kind {kind}")
    return f"memcpy {label}"


def _sync_name(sync_type, labels):
    if sync_type in labels:
        return labels[sync_type]
    return SYNC_TYPE.get(sync_type, f"synchronize type {sync_type}")


def _lookup_string(value, strings):
    if value is None:
        return "unresolved"
    if isinstance(value, str):
        return value
    key = int(value)
    if key not in strings:
        return f"unresolved:{key}"
    return strings[key]


def _string_ids(connection):
    if not _has_table(connection, "StringIds"):
        raise StepceilError("sqlite export is missing table StringIds")
    _require_columns(connection, "StringIds", ("id", "value"))
    strings = {}
    for row in connection.execute("SELECT id, value FROM StringIds"):
        text = "" if row["value"] is None else str(row["value"])
        strings[int(row["id"])] = text
    return strings


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

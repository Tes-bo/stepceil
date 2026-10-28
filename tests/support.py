"""Build a small nsys-shaped sqlite export for tests."""

import sqlite3
from pathlib import Path


def write_export(
    path, ranges, kernels=(), copies=(), memsets=(), syncs=(), runtime=(),
):
    """runtime rows are host API calls. That table is written only when given."""
    path = Path(path)
    connection = sqlite3.connect(path)
    try:
        connection.execute(
            "CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT NOT NULL)"
        )
        connection.execute(
            """
            CREATE TABLE NVTX_EVENTS (
                start INTEGER,
                end INTEGER,
                eventType INTEGER,
                text TEXT,
                textId INTEGER
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (
                start INTEGER,
                end INTEGER,
                deviceId INTEGER,
                streamId INTEGER,
                contextId INTEGER,
                shortName INTEGER
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY (
                start INTEGER,
                end INTEGER,
                deviceId INTEGER,
                streamId INTEGER,
                contextId INTEGER,
                copyKind INTEGER
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE CUPTI_ACTIVITY_KIND_MEMSET (
                start INTEGER,
                end INTEGER,
                deviceId INTEGER,
                streamId INTEGER,
                contextId INTEGER
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE CUPTI_ACTIVITY_KIND_SYNCHRONIZATION (
                start INTEGER,
                end INTEGER,
                syncType INTEGER
            )
            """
        )
        next_id = 1

        def intern(text):
            nonlocal next_id
            while connection.execute(
                "SELECT 1 FROM StringIds WHERE id = ?",
                (next_id,),
            ).fetchone():
                next_id += 1
            connection.execute(
                "INSERT INTO StringIds (id, value) VALUES (?, ?)",
                (next_id, text),
            )
            assigned = next_id
            next_id += 1
            return assigned

        for item in ranges:
            event_type = item.get("event_type", 59)
            if item.get("text_id") is not None:
                text_id = item["text_id"]
                connection.execute(
                    "INSERT INTO StringIds (id, value) VALUES (?, ?)",
                    (text_id, item["text_id_value"]),
                )
                next_id = max(next_id, text_id + 1)
                connection.execute(
                    """
                    INSERT INTO NVTX_EVENTS
                        (start, end, eventType, text, textId)
                    VALUES (?, ?, ?, NULL, ?)
                    """,
                    (item["start"], item["end"], event_type, text_id),
                )
            else:
                connection.execute(
                    """
                    INSERT INTO NVTX_EVENTS
                        (start, end, eventType, text, textId)
                    VALUES (?, ?, ?, ?, NULL)
                    """,
                    (item["start"], item["end"], event_type, item.get("text")),
                )
        for item in kernels:
            connection.execute(
                """
                INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL
                    (start, end, deviceId, streamId, contextId, shortName)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    item["start"],
                    item["end"],
                    item.get("device_id", 0),
                    item.get("stream_id", 1),
                    item.get("context_id", 1),
                    intern(item["name"]),
                ),
            )
        for item in copies:
            connection.execute(
                """
                INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY
                    (start, end, deviceId, streamId, contextId, copyKind)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    item["start"],
                    item["end"],
                    item.get("device_id", 0),
                    item.get("stream_id", 1),
                    item.get("context_id", 1),
                    item.get("copy_kind", 1),
                ),
            )
        for item in memsets:
            connection.execute(
                """
                INSERT INTO CUPTI_ACTIVITY_KIND_MEMSET
                    (start, end, deviceId, streamId, contextId)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    item["start"],
                    item["end"],
                    item.get("device_id", 0),
                    item.get("stream_id", 1),
                    item.get("context_id", 1),
                ),
            )
        for item in syncs:
            connection.execute(
                """
                INSERT INTO CUPTI_ACTIVITY_KIND_SYNCHRONIZATION
                    (start, end, syncType)
                VALUES (?, ?, ?)
                """,
                (item["start"], item["end"], item["sync_type"]),
            )
        if runtime:
            connection.execute(
                """
                CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME (
                    start INTEGER,
                    end INTEGER,
                    correlationId INTEGER,
                    nameId INTEGER
                )
                """
            )
        for index, item in enumerate(runtime, start=1):
            connection.execute(
                """
                INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME
                    (start, end, correlationId, nameId)
                VALUES (?, ?, ?, ?)
                """,
                (item["start"], item["end"], index, intern(item["name"])),
            )
        connection.commit()
    finally:
        connection.close()

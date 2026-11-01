"""Copy the tables stepceil reads from an nsys sqlite export.

    python3 tests/data/trim_export.py REPORT.sqlite OUT.sqlite

Each table keeps the CREATE TABLE statement Nsight Systems wrote, so the
copy has the real column names, types, and order. StringIds keeps only the
strings those rows reference, found from the `REFERENCES StringIds(id)`
comments in that schema. Every other table, such as the target system,
the environment, and the process list, is left out.
"""

import re
import sqlite3
import sys
from pathlib import Path

TABLES = (
    "NVTX_EVENTS",
    "CUPTI_ACTIVITY_KIND_KERNEL",
    "CUPTI_ACTIVITY_KIND_MEMCPY",
    "CUPTI_ACTIVITY_KIND_MEMSET",
    "CUPTI_ACTIVITY_KIND_SYNCHRONIZATION",
    "CUPTI_ACTIVITY_KIND_RUNTIME",
    "ENUM_CUDA_MEMCPY_OPER",
    "ENUM_CUPTI_SYNC_TYPE",
)
STRING_COLUMN = re.compile(r"^\s*(\w+)\s+.*REFERENCES StringIds\(id\)", re.M)


def trim(source, target):
    target = Path(target)
    if target.exists():
        raise SystemExit(f"{target} exists")
    src = sqlite3.connect(f"file:{Path(source).resolve()}?mode=ro", uri=True)
    dst = sqlite3.connect(target)
    try:
        string_ids = set()
        for table in (*TABLES, "StringIds"):
            row = src.execute(
                "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?",
                (table,),
            ).fetchone()
            if row is None:
                continue
            dst.execute(row[0])
            if table == "StringIds":
                continue
            rows = src.execute(f"SELECT * FROM {table}").fetchall()
            if rows:
                marks = ", ".join("?" * len(rows[0]))
                dst.executemany(f"INSERT INTO {table} VALUES ({marks})", rows)
            for column in STRING_COLUMN.findall(row[0]):
                string_ids.update(
                    value for (value,) in src.execute(
                        f"SELECT DISTINCT {column} FROM {table} "
                        f"WHERE {column} IS NOT NULL"
                    )
                )
        dst.executemany(
            "INSERT INTO StringIds (id, value) VALUES (?, ?)",
            (
                row for row in src.execute("SELECT id, value FROM StringIds")
                if row[0] in string_ids
            ),
        )
        dst.commit()
        dst.execute("VACUUM")
    finally:
        src.close()
        dst.close()


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    trim(sys.argv[1], sys.argv[2])

"""Write the sqlite files shown in the README.

These are synthetic nsys-shaped exports, not captured reports.
Run from the repository root: python3 examples/build_examples.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))

from support import write_export  # noqa: E402


def main():
    out = Path(__file__).resolve().parent
    for name in ("sync.sqlite", "after.sqlite", "overlap.sqlite", "names.sqlite"):
        (out / name).unlink(missing_ok=True)
    write_export(
        out / "sync.sqlite",
        [{"text": "step", "start": 0, "end": 1000}],
        kernels=[{"name": "pressure", "start": 0, "end": 80}],
        syncs=[{"sync_type": 4, "start": 0, "end": 1000}],
    )
    write_export(
        out / "after.sqlite",
        [{"text": "step", "start": 0, "end": 200}],
        kernels=[{"name": "pressure", "start": 0, "end": 80}],
        syncs=[{"sync_type": 4, "start": 0, "end": 200}],
    )
    write_export(
        out / "overlap.sqlite",
        [{"text": "step", "start": 0, "end": 100}],
        kernels=[
            {"name": "alpha", "start": 0, "end": 60, "stream_id": 1},
            {"name": "beta", "start": 40, "end": 100, "stream_id": 2},
        ],
    )
    write_export(
        out / "names.sqlite",
        [
            {"text": "inner", "start": 0, "end": 4},
            {"text": "inner", "start": 4, "end": 8},
            {"text": "inner", "start": 8, "end": 12},
            {"text": "inner", "start": 12, "end": 16},
            {"text": "step", "start": 0, "end": 16},
        ],
    )


if __name__ == "__main__":
    main()

import _path  # noqa: F401
import sqlite3
import tempfile
import unittest
from pathlib import Path

from stepceil.diff import compare_ledgers
from stepceil.errors import StepceilError
from stepceil.format import render_diff_text, render_ranges_text, render_text
from stepceil.ledger import account
from stepceil.model import DeviceOp, StepWindow, SyncOp
from stepceil.nsys import list_ranges, load_path
from _path import ROOT
from support import write_export


class ReadTest(unittest.TestCase):
    def test_sqlite_matches_the_ledger(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "report.sqlite"
            write_export(
                path,
                [{"text": "step", "start": 0, "end": 1000}],
                kernels=[{"name": "pressure", "start": 0, "end": 80}],
                syncs=[{"sync_type": 4, "start": 0, "end": 1000}],
            )
            loaded = load_path(path, "step")
            direct = account(StepWindow(
                "step", 0, 1000,
                (DeviceOp("kernel", "pressure", 0, 80, 1, 0, 1),),
                (SyncOp("context synchronize", 0, 1000),),
            ))
            self.assertEqual(account(loaded), direct)

    def test_step_range_is_not_a_blocker(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "report.sqlite"
            write_export(
                path,
                [
                    {"text": "step", "start": 0, "end": 100},
                    {"text": "phase", "start": 10, "end": 90},
                ],
            )
            loaded = load_path(path, "step")
            self.assertEqual([op.name for op in loaded.nvtx_ops], ["phase"])
            ledger = account(loaded)
            self.assertEqual(ledger["longest_blocker"]["kind"], "nvtx")
            self.assertEqual(ledger["longest_blocker"]["name"], "phase")
            self.assertEqual(ledger["longest_blocker"]["blocking_ns"], 80)

    def test_memcpy_memset_and_other_range_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "report.sqlite"
            write_export(
                path,
                [
                    {"text": "other", "start": 0, "end": 5},
                    {"text": "step", "start": 0, "end": 100},
                    {"text": "step", "start": 0, "end": None, "event_type": 34},
                ],
                kernels=[{"name": "pressure", "start": 0, "end": 10}],
                copies=[{
                    "start": 10,
                    "end": 70,
                    "copy_kind": 1,
                }],
                memsets=[{"start": 70, "end": 90}],
            )
            ledger = account(load_path(path, "step"))
            self.assertEqual(ledger["longest_blocker"]["kind"], "memcpy")
            self.assertEqual(ledger["longest_blocker"]["name"], "memcpy HtoD")
            self.assertEqual(ledger["ceiling_ns"], 10)

    def test_registered_nvtx_string_and_text_override(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "report.sqlite"
            write_export(
                path,
                [{
                    "text_id": 7,
                    "text_id_value": "step",
                    "start": 0,
                    "end": 40,
                }],
                kernels=[{"name": "pressure", "start": 0, "end": 40}],
            )
            self.assertEqual(account(load_path(path, "step"))["ceiling_ns"], 40)

            override = Path(tmp) / "override.sqlite"
            write_export(
                override,
                [{
                    "text_id": 7,
                    "text_id_value": "step",
                    "start": 0,
                    "end": 40,
                }],
            )
            connection = sqlite3.connect(override)
            connection.execute("UPDATE NVTX_EVENTS SET text = 'other'")
            connection.commit()
            connection.close()
            with self.assertRaisesRegex(StepceilError, "found 0"):
                load_path(override, "step")
            self.assertEqual(load_path(override, "other").end_ns, 40)

    def test_start_end_range_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "report.sqlite"
            write_export(
                path,
                [{"text": "step", "start": 0, "end": 25, "event_type": 60}],
                kernels=[{"name": "pressure", "start": 0, "end": 25}],
            )
            self.assertEqual(account(load_path(path, "step"))["ceiling_ns"], 25)

    def test_range_count_and_missing_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            two = Path(tmp) / "two.sqlite"
            write_export(two, [
                {"text": "step", "start": 0, "end": 10},
                {"text": "step", "start": 20, "end": None},
            ])
            with self.assertRaisesRegex(StepceilError, "found 2"):
                load_path(two, "step")
            none = Path(tmp) / "none.sqlite"
            write_export(none, [{"text": "other", "start": 0, "end": 10}])
            with self.assertRaisesRegex(StepceilError, "found 0"):
                load_path(none, "step")
            open_ended = Path(tmp) / "open.sqlite"
            write_export(open_ended, [{"text": "step", "start": 0, "end": None}])
            with self.assertRaisesRegex(StepceilError, "no end timestamp"):
                load_path(open_ended, "step")
            listed = list_ranges(two)
            self.assertEqual(listed["ranges"], [
                {
                    "name": "step",
                    "count": 2,
                    "closed": 1,
                    "shortest_ns": 10,
                    "longest_ns": 10,
                },
            ])
            self.assertEqual(list_ranges(open_ended)["ranges"], [
                {
                    "name": "step",
                    "count": 1,
                    "closed": 0,
                    "shortest_ns": None,
                    "longest_ns": None,
                },
            ])

    def test_demangled_name_is_used_when_short_name_is_absent(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "report.sqlite"
            connection = sqlite3.connect(path)
            connection.execute(
                "CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT NOT NULL)"
            )
            connection.execute(
                """
                CREATE TABLE NVTX_EVENTS (
                    start INTEGER, end INTEGER, eventType INTEGER, text TEXT
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (
                    start INTEGER, end INTEGER, deviceId INTEGER,
                    streamId INTEGER, demangledName INTEGER
                )
                """
            )
            connection.execute("INSERT INTO StringIds VALUES (3, 'pressure')")
            connection.execute(
                "INSERT INTO NVTX_EVENTS VALUES (0, 30, 59, 'step')"
            )
            connection.execute(
                """
                INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL
                VALUES (0, 30, 0, 1, 3)
                """
            )
            connection.commit()
            connection.close()
            ledger = account(load_path(path, "step"))
            self.assertEqual(ledger["hottest_kernel"]["name"], "pressure")
            self.assertEqual(ledger["ceiling_ns"], 30)

    def test_bad_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "missing.sqlite"
            with self.assertRaisesRegex(StepceilError, "no such file"):
                load_path(missing, "step")
            junk = Path(tmp) / "junk.sqlite"
            junk.write_bytes(b"not sqlite")
            with self.assertRaisesRegex(StepceilError, "not an nsys sqlite export"):
                load_path(junk, "step")
            with self.assertRaisesRegex(StepceilError, "empty range name"):
                load_path(junk, "")

    def test_export_enum_labels_override_the_builtin_copy_kinds(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "report.sqlite"
            write_export(
                path,
                [{"text": "step", "start": 0, "end": 40}],
                copies=[{"start": 0, "end": 40, "copy_kind": 3}],
                syncs=[{"sync_type": 4, "start": 0, "end": 40}],
            )
            connection = sqlite3.connect(path)
            connection.execute(
                """
                CREATE TABLE ENUM_CUDA_MEMCPY_OPER (
                    id INTEGER, name TEXT, label TEXT
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE ENUM_CUPTI_SYNC_TYPE (
                    id INTEGER, name TEXT, label TEXT
                )
                """
            )
            # Kind 3 is HtoH in older CUPTI headers. This export says otherwise.
            connection.execute(
                "INSERT INTO ENUM_CUDA_MEMCPY_OPER VALUES (3, 'X', 'Host-to-Array')"
            )
            connection.execute(
                "INSERT INTO ENUM_CUPTI_SYNC_TYPE VALUES (4, 'Y', 'Context sync')"
            )
            connection.commit()
            connection.close()
            ledger = account(load_path(path, "step"))
            self.assertEqual(ledger["longest_blocker"]["name"], "memcpy Host-to-Array")
            self.assertNotIn("HtoH", ledger["longest_blocker"]["name"])

    def test_peer_copy_names_the_other_gpu(self):
        with tempfile.TemporaryDirectory() as tmp:
            touched = Path(tmp) / "peer.sqlite"
            write_export(
                touched,
                [{"text": "step", "start": 0, "end": 40}],
                copies=[{"start": 0, "end": 40, "copy_kind": 10}],
            )
            _add_peer_columns(touched, src=0, dst=1)
            with self.assertRaisesRegex(StepceilError, "device ids 0, 1"):
                account(load_path(touched, "step"))

            outside = Path(tmp) / "outside.sqlite"
            write_export(
                outside,
                [{"text": "step", "start": 0, "end": 40}],
                kernels=[{"name": "pressure", "start": 0, "end": 40}],
                copies=[{"start": 100, "end": 140, "copy_kind": 10}],
            )
            _add_peer_columns(outside, src=0, dst=1)
            ledger = account(load_path(outside, "step"))
            self.assertEqual(ledger["hottest_kernel"]["name"], "pressure")

            same = Path(tmp) / "same.sqlite"
            write_export(
                same,
                [{"text": "step", "start": 0, "end": 40}],
                copies=[{"start": 0, "end": 40, "copy_kind": 4}],
            )
            _add_peer_columns(same, src=0, dst=0)
            ledger = account(load_path(same, "step"))
            self.assertEqual(ledger["longest_blocker"]["kind"], "memcpy")

    def test_global_pid_is_the_process(self):
        with tempfile.TemporaryDirectory() as tmp:
            mixed = Path(tmp) / "mixed.sqlite"
            write_export(
                mixed,
                [{"text": "step", "start": 0, "end": 40}],
                kernels=[
                    {"name": "pressure", "start": 0, "end": 10, "stream_id": 7},
                    {"name": "other", "start": 0, "end": 10, "stream_id": 7},
                ],
            )
            _assign_global_pids(
                mixed, "CUPTI_ACTIVITY_KIND_KERNEL", (10, 11),
            )
            with self.assertRaisesRegex(StepceilError, "process ids 10, 11"):
                account(load_path(mixed, "step"))

            outside = Path(tmp) / "outside.sqlite"
            write_export(
                outside,
                [{"text": "step", "start": 0, "end": 40}],
                kernels=[
                    {"name": "pressure", "start": 0, "end": 40},
                    {"name": "other", "start": 100, "end": 140},
                ],
            )
            _assign_global_pids(
                outside, "CUPTI_ACTIVITY_KIND_KERNEL", (10, 11),
            )
            ledger = account(load_path(outside, "step"))
            self.assertEqual(ledger["hottest_kernel"]["name"], "pressure")
            self.assertEqual(ledger["ceiling_ns"], 40)

    def test_copy_and_memset_blockers_carry_their_byte_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            copied = Path(tmp) / "copy.sqlite"
            write_export(
                copied,
                [{"text": "step", "start": 0, "end": 80}],
                kernels=[{"name": "pressure", "start": 0, "end": 10}],
                copies=[{"start": 10, "end": 80, "copy_kind": 1}],
            )
            _set_bytes(copied, "CUPTI_ACTIVITY_KIND_MEMCPY", 4096)
            ledger = account(load_path(copied, "step"))
            self.assertEqual(ledger["longest_blocker"]["kind"], "memcpy")
            self.assertEqual(ledger["longest_blocker"]["bytes"], 4096)
            self.assertEqual(ledger["decision"], "change memcpy HtoD first")

            kernel_led = account(load_path(
                _kernel_only(Path(tmp) / "kernel.sqlite"), "step"
            ))
            self.assertEqual(kernel_led["longest_blocker"]["kind"], "kernel")
            self.assertIsNone(kernel_led["longest_blocker"]["bytes"])

            cleared = Path(tmp) / "memset.sqlite"
            write_export(
                cleared,
                [{"text": "step", "start": 0, "end": 20}],
                memsets=[{"start": 0, "end": 20}],
            )
            _set_bytes(cleared, "CUPTI_ACTIVITY_KIND_MEMSET", 16)
            self.assertEqual(
                account(load_path(cleared, "step"))["longest_blocker"]["bytes"],
                16,
            )

            omitted = Path(tmp) / "omitted.sqlite"
            write_export(
                omitted,
                [{"text": "step", "start": 0, "end": 20}],
                copies=[{"start": 0, "end": 20, "copy_kind": 1}],
            )
            self.assertIsNone(
                account(load_path(omitted, "step"))["longest_blocker"]["bytes"]
            )

    def test_runtime_call_names_an_unknown_synchronize(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "report.sqlite"
            _export_sync_runtime(path, "Unknown", [
                (200, 210, "cudaLaunchKernel"),
                (0, 100, "cudaDeviceSynchronize"),
            ])
            ledger = account(load_path(path, "step"))
            self.assertEqual(
                ledger["longest_blocker"]["name"], "cudaDeviceSynchronize"
            )

    def test_a_launch_does_not_rename_the_synchronize(self):
        with tempfile.TemporaryDirectory() as tmp:
            overlap = Path(tmp) / "overlap.sqlite"
            _export_sync_runtime(overlap, "Unknown", [
                (0, 50, "cudaLaunchKernel"),
            ])
            self.assertEqual(
                account(load_path(overlap, "step"))["longest_blocker"]["name"],
                "Unknown",
            )
            covering = Path(tmp) / "covering.sqlite"
            _export_sync_runtime(covering, "Unknown", [
                (0, 100, "cudaLaunchKernel"),
            ])
            self.assertEqual(
                account(load_path(covering, "step"))["longest_blocker"]["name"],
                "Unknown",
            )
            named = Path(tmp) / "named.sqlite"
            _export_sync_runtime(named, "Stream wait sync", [
                (0, 40, "cudaLaunchKernel"),
                (0, 100, "cudaStreamSynchronize"),
            ])
            self.assertEqual(
                account(load_path(named, "step"))["longest_blocker"]["name"],
                "cudaStreamSynchronize",
            )
            both = Path(tmp) / "both.sqlite"
            _export_sync_runtime(both, "Stream wait sync", [
                (0, 100, "cudaDeviceSynchronize"),
                (0, 100, "cudaStreamSynchronize"),
            ])
            self.assertEqual(
                account(load_path(both, "step"))["longest_blocker"]["name"],
                "Stream wait sync",
            )

    def test_runtime_from_another_process_does_not_rename(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "proc.sqlite"
            _export_sync_runtime(path, "Unknown", [
                (0, 100, "cudaDeviceSynchronize"),
                (0, 100, "cudaStreamSynchronize"),
            ])
            same = 10 << 24
            other = 11 << 24
            connection = sqlite3.connect(path)
            connection.execute(
                "ALTER TABLE CUPTI_ACTIVITY_KIND_SYNCHRONIZATION "
                "ADD COLUMN globalPid INTEGER"
            )
            connection.execute(
                "UPDATE CUPTI_ACTIVITY_KIND_SYNCHRONIZATION SET globalPid = ?",
                (same,),
            )
            connection.execute(
                "ALTER TABLE CUPTI_ACTIVITY_KIND_RUNTIME ADD COLUMN globalTid INTEGER"
            )
            rows = connection.execute(
                "SELECT rowid, nameId FROM CUPTI_ACTIVITY_KIND_RUNTIME ORDER BY rowid"
            ).fetchall()
            for rowid, name_id in rows:
                value = connection.execute(
                    "SELECT value FROM StringIds WHERE id = ?",
                    (name_id,),
                ).fetchone()[0]
                process = same if value == "cudaDeviceSynchronize" else other
                connection.execute(
                    "UPDATE CUPTI_ACTIVITY_KIND_RUNTIME SET globalTid = ? "
                    "WHERE rowid = ?",
                    (process | 7, rowid),
                )
            connection.commit()
            connection.close()
            self.assertEqual(
                account(load_path(path, "step"))["longest_blocker"]["name"],
                "cudaDeviceSynchronize",
            )

    def test_step_range_names_its_process(self):
        own = 10 << 24
        other = 11 << 24
        cases = (
            ("own", own, own, "runtime"),
            ("foreign_call", own, other, "gap"),
            ("unnamed_kernel", None, other, "gap"),
        )
        with tempfile.TemporaryDirectory() as tmp:
            for label, kernel_pid, call_pid, kind in cases:
                path = Path(tmp) / f"{label}.sqlite"
                _export_step_process(path, own | 5, kernel_pid, call_pid | 3)
                loaded = load_path(path, "step")
                self.assertEqual(loaded.process_id, own)
                ledger = account(loaded)
                self.assertEqual(ledger["longest_blocker"]["kind"], kind, label)

            foreign = Path(tmp) / "foreign.sqlite"
            _export_step_process(foreign, own | 5, other, other | 3)
            with self.assertRaisesRegex(
                StepceilError, f"process ids {own}, {other}",
            ):
                account(load_path(foreign, "step"))

    def test_kernel_name_falls_back_on_each_row(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "report.sqlite"
            connection = sqlite3.connect(path)
            connection.execute(
                "CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT NOT NULL)"
            )
            connection.execute(
                """
                CREATE TABLE NVTX_EVENTS (
                    start INTEGER, end INTEGER, eventType INTEGER, text TEXT
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (
                    start INTEGER, end INTEGER, deviceId INTEGER,
                    streamId INTEGER, shortName INTEGER,
                    demangledName INTEGER, mangledName INTEGER
                )
                """
            )
            connection.execute("INSERT INTO StringIds VALUES (3, 'void sum<int>()')")
            connection.execute("INSERT INTO StringIds VALUES (4, '_Z3sumIiEv')")
            connection.execute("INSERT INTO NVTX_EVENTS VALUES (0, 30, 59, 'step')")
            # shortName is null, and id 99 is not in StringIds.
            connection.execute(
                "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (0, 20, 0, 1, NULL, 3, 99)"
            )
            connection.execute(
                "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (20, 30, 0, 1, 99, NULL, 4)"
            )
            connection.commit()
            connection.close()
            ops = load_path(path, "step").device_ops
            self.assertEqual(
                [(op.name, op.symbol) for op in ops],
                [
                    ("void sum<int>()", "void sum<int>()"),
                    ("_Z3sumIiEv", "_Z3sumIiEv"),
                ],
            )

    def test_step_without_a_start_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "report.sqlite"
            write_export(path, [{"text": "step", "start": None, "end": 10}])
            with self.assertRaisesRegex(StepceilError, "no start timestamp"):
                load_path(path, "step")

    def test_occurrence_picks_one_closed_range_by_start(self):
        ranges = [
            {"text": "step", "start": 200, "end": 300},
            {"text": "step", "start": 50, "end": None},
            {"text": "step", "start": 0, "end": 100},
            {"text": "step", "start": 100, "end": 200},
            {"text": "step", "start": 150, "end": 150},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            forward = Path(tmp) / "forward.sqlite"
            backward = Path(tmp) / "backward.sqlite"
            write_export(forward, ranges)
            write_export(backward, list(reversed(ranges)))
            for path in (forward, backward):
                windows = [load_path(path, "step", n) for n in (1, 2, 3)]
                self.assertEqual(
                    [(w.start_ns, w.end_ns, w.occurrence) for w in windows],
                    [(0, 100, 1), (100, 200, 2), (200, 300, 3)],
                )
                with self.assertRaisesRegex(StepceilError, "found 5; choose one"):
                    load_path(path, "step")
                with self.assertRaisesRegex(
                    StepceilError, "has 3 closed occurrences; occurrence 4",
                ):
                    load_path(path, "step", 4)
                for bad in (0, -1, True, 1.0, "2"):
                    with self.assertRaisesRegex(StepceilError, "at least 1"):
                        load_path(path, "step", bad)
            self.assertEqual(account(load_path(forward, "step", 2))["occurrence"], 2)

            single = Path(tmp) / "single.sqlite"
            write_export(single, [{"text": "step", "start": 0, "end": 10}])
            self.assertEqual(load_path(single, "step").occurrence, 1)
            self.assertEqual(
                account(load_path(single, "step", 1)),
                account(load_path(single, "step")),
            )

    def test_other_occurrences_are_not_blockers(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "report.sqlite"
            write_export(path, [
                {"text": "step", "start": 0, "end": 100},
                {"text": "step", "start": 60, "end": 200, "event_type": 60},
                {"text": "phase", "start": 10, "end": 40},
            ])
            loaded = load_path(path, "step", 1)
            self.assertEqual([op.name for op in loaded.nvtx_ops], ["phase"])
            ledger = account(loaded)
            self.assertEqual(ledger["longest_blocker"]["kind"], "gap")
            self.assertEqual(ledger["longest_blocker"]["blocking_ns"], 60)

    def test_diff_keeps_one_occurrence(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "loop.sqlite"
            write_export(path, [
                {"text": "step", "start": 0, "end": 100},
                {"text": "step", "start": 100, "end": 150},
            ])
            first = account(load_path(path, "step", 1))
            second = account(load_path(path, "step", 2))
            report = compare_ledgers(first, first)
            self.assertEqual(report["occurrence"], 1)
            with self.assertRaisesRegex(StepceilError, "occurrences differ: 1 and 2"):
                compare_ledgers(first, second)

    def test_readme_loop_accounts_a_later_occurrence(self):
        readme = (ROOT / "README.md").read_text()
        loop = ROOT / "examples" / "loop.sqlite"
        listed = list_ranges(loop)["ranges"]
        self.assertEqual([(r["name"], r["count"], r["closed"]) for r in listed], [
            ("step", 3, 3),
        ])
        decisions = [
            account(load_path(loop, "step", n))["decision"] for n in (1, 2, 3)
        ]
        self.assertEqual(decisions, [
            "change cudaMalloc_v3020 first",
            "rewrite pressure",
            "rewrite pressure",
        ])
        second = render_text(account(load_path(loop, "step", 2)))
        excerpt = readme.split("--occurrence 2\n```\n\n```text\n", 1)[1]
        excerpt = excerpt.split("```", 1)[0]
        self.assertTrue(second.startswith(excerpt), excerpt)

    def test_readme_examples_are_the_exports_in_the_repository(self):
        readme = (ROOT / "README.md").read_text()
        examples = ROOT / "examples"
        sync = account(load_path(examples / "sync.sqlite", "step"))
        overlap = account(load_path(examples / "overlap.sqlite", "step"))
        after = account(load_path(examples / "after.sqlite", "step"))
        diff = compare_ledgers(sync, after)
        self.assertIn(render_text(sync), readme)
        self.assertIn(render_text(overlap), readme)
        self.assertIn(render_diff_text(diff), readme)
        self.assertIn(render_ranges_text(list_ranges(examples / "names.sqlite")), readme)
        self.assertTrue(diff["faster"])
        self.assertEqual(diff["step_time_ns_delta"], -800)
        self.assertEqual(diff["after_longest_blocker"]["kind"], "sync")


def _kernel_only(path):
    write_export(
        path,
        [{"text": "step", "start": 0, "end": 40}],
        kernels=[{"name": "pressure", "start": 0, "end": 40}],
    )
    return path


def _set_bytes(path, table, size):
    connection = sqlite3.connect(path)
    connection.execute(f"ALTER TABLE {table} ADD COLUMN bytes INTEGER")
    connection.execute(f"UPDATE {table} SET bytes = ?", (size,))
    connection.commit()
    connection.close()


def _export_sync_runtime(path, label, calls):
    write_export(
        path,
        [{"text": "step", "start": 0, "end": 100}],
        syncs=[{"sync_type": 0, "start": 0, "end": 100}],
    )
    connection = sqlite3.connect(path)
    connection.execute(
        "ALTER TABLE CUPTI_ACTIVITY_KIND_SYNCHRONIZATION "
        "ADD COLUMN correlationId INTEGER"
    )
    connection.execute(
        "UPDATE CUPTI_ACTIVITY_KIND_SYNCHRONIZATION SET correlationId = 9"
    )
    connection.execute(
        "CREATE TABLE ENUM_CUPTI_SYNC_TYPE (id INTEGER, name TEXT, label TEXT)"
    )
    connection.execute(
        "INSERT INTO ENUM_CUPTI_SYNC_TYPE VALUES (0, 'ENUM', ?)",
        (label,),
    )
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
    for index, (start, end, name) in enumerate(calls, start=3):
        connection.execute(
            "INSERT INTO StringIds (id, value) VALUES (?, ?)",
            (index, name),
        )
        connection.execute(
            "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?, ?, 9, ?)",
            (start, end, index),
        )
    connection.commit()
    connection.close()


def _export_step_process(path, step_tid, kernel_pid, call_tid):
    """A 100 ns step, a 20 ns kernel, and a host call on the idle after it."""
    write_export(
        path,
        [{"text": "step", "start": 0, "end": 100}],
        kernels=[{"name": "pressure", "start": 0, "end": 20}],
    )
    connection = sqlite3.connect(path)
    connection.execute("ALTER TABLE NVTX_EVENTS ADD COLUMN globalTid INTEGER")
    connection.execute("UPDATE NVTX_EVENTS SET globalTid = ?", (step_tid,))
    connection.execute(
        "ALTER TABLE CUPTI_ACTIVITY_KIND_KERNEL ADD COLUMN globalPid INTEGER"
    )
    connection.execute(
        "UPDATE CUPTI_ACTIVITY_KIND_KERNEL SET globalPid = ?", (kernel_pid,),
    )
    connection.execute(
        """
        CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME (
            start INTEGER, end INTEGER, globalTid INTEGER,
            correlationId INTEGER, nameId INTEGER
        )
        """
    )
    connection.execute(
        "INSERT INTO StringIds (id, value) VALUES (900, 'cudaMallocHost_v3020')"
    )
    connection.execute(
        "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (20, 90, ?, 1, 900)",
        (call_tid,),
    )
    connection.commit()
    connection.close()


def _assign_global_pids(path, table, pids):
    connection = sqlite3.connect(path)
    connection.execute(f"ALTER TABLE {table} ADD COLUMN globalPid INTEGER")
    rows = connection.execute(
        f"SELECT rowid FROM {table} ORDER BY rowid"
    ).fetchall()
    for row, pid in zip(rows, pids):
        connection.execute(
            f"UPDATE {table} SET globalPid = ? WHERE rowid = ?",
            (pid, row[0]),
        )
    connection.commit()
    connection.close()


def _add_peer_columns(path, src, dst):
    connection = sqlite3.connect(path)
    connection.execute(
        "ALTER TABLE CUPTI_ACTIVITY_KIND_MEMCPY ADD COLUMN srcDeviceId INTEGER"
    )
    connection.execute(
        "ALTER TABLE CUPTI_ACTIVITY_KIND_MEMCPY ADD COLUMN dstDeviceId INTEGER"
    )
    connection.execute(
        "UPDATE CUPTI_ACTIVITY_KIND_MEMCPY SET srcDeviceId = ?, dstDeviceId = ?",
        (src, dst),
    )
    connection.commit()
    connection.close()


if __name__ == "__main__":
    unittest.main()

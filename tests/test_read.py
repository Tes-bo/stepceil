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

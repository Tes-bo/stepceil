"""One real capture, so the reader meets the schema Nsight Systems writes.

tests/data/nsys-2024.6-loop.sqlite is tests/data/loop.cu on a Tesla V100,
captured with Nsight Systems 2024.6.1 and trimmed by trim_export.py:

    nvcc -O2 -arch=sm_70 loop.cu -o loop
    nsys profile --trace=cuda,nvtx -o loop ./loop
    nsys export --type sqlite -o loop.sqlite loop.nsys-rep

The GPU was shared with other processes during the capture. In the later
iterations pressure starts milliseconds after its launch returns, so the
synchronize covers that wait. The numbers below are that capture's.
"""

import _path  # noqa: F401
import unittest

from stepceil.ledger import ASSUMPTION_EXCLUSIVE, account
from stepceil.nsys import list_ranges, load_path
from _path import ROOT

EXPORT = ROOT / "tests" / "data" / "nsys-2024.6-loop.sqlite"
SYNC = "cudaDeviceSynchronize_v3020"


class RealExportTest(unittest.TestCase):
    def test_ranges(self):
        self.assertEqual(list_ranges(EXPORT)["ranges"], [
            {
                "name": "forward",
                "count": 5,
                "closed": 5,
                "shortest_ns": 11192,
                "longest_ns": 377317,
            },
            {
                "name": "step",
                "count": 5,
                "closed": 5,
                "shortest_ns": 2919721,
                "longest_ns": 583645188,
            },
        ])

    def test_every_iteration_is_one_process_and_one_stream(self):
        for occurrence in range(1, 6):
            window = load_path(EXPORT, "step", occurrence)
            self.assertIsNotNone(window.process_id)
            named = (*window.device_ops, *window.syncs, *window.runtime_ops)
            self.assertEqual(
                {item.process_id for item in named}, {window.process_id},
            )
            kernels = [op for op in window.device_ops if op.kind == "kernel"]
            self.assertEqual(
                [(op.name, op.symbol) for op in kernels],
                [("pressure", "_Z8pressurePfii"), ("tail", "_Z4tailPfi")],
            )
            ledger = account(window)
            self.assertEqual(ledger["assumption"], ASSUMPTION_EXCLUSIVE)
            self.assertEqual(
                ledger["gpu_busy_ns"] + ledger["gpu_idle_ns"],
                ledger["step_time_ns"],
            )
            self.assertEqual(ledger["hottest_kernel"]["symbol"], "_Z8pressurePfii")

    def test_first_iteration_is_setup(self):
        ledger = account(load_path(EXPORT, "step", 1))
        self.assertEqual(ledger["step_time_ns"], 583645188)
        self.assertEqual(ledger["decision"], "change gpu idle first")
        self.assertEqual(ledger["separate"]["kind"], "runtime")
        self.assertEqual(ledger["separate"]["name"], "cudaMalloc_v3020")
        self.assertEqual(ledger["separate"]["blocking_ns"], 188840932)

    def test_the_synchronize_is_one_blocker(self):
        # The runtime call is wider than the synchronization record. It is
        # the synchronize, not a second, runtime blocker around it.
        times = []
        for occurrence in range(2, 6):
            window = load_path(EXPORT, "step", occurrence)
            self.assertEqual([sync.name for sync in window.syncs], [SYNC])
            self.assertNotIn(SYNC, [call.name for call in window.runtime_ops])
            ledger = account(window)
            self.assertEqual(ledger["longest_blocker"]["kind"], "sync")
            self.assertEqual(ledger["longest_blocker"]["name"], SYNC)
            self.assertIsNone(ledger["contained"])
            self.assertEqual(ledger["separate"]["name"], "pressure")
            self.assertEqual(ledger["decision"], f"change {SYNC} first")
            times.append(ledger["step_time_ns"])
        self.assertEqual(times, [3999049, 2923571, 2934095, 2919721])


if __name__ == "__main__":
    unittest.main()

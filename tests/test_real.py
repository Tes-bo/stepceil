"""Real captures, so the reader meets the schema Nsight Systems writes.

tests/data/nsys-2024.6-loop.sqlite is tests/data/loop.cu on a Tesla V100,
captured with Nsight Systems 2024.6.1 and trimmed by trim_export.py:

    nvcc -O2 -arch=sm_70 loop.cu -o loop
    nsys profile --trace=cuda,nvtx -o loop ./loop
    nsys export --type sqlite -o loop.sqlite loop.nsys-rep

The GPU was shared with other processes during the capture. In the later
iterations pressure starts milliseconds after its launch returns, and the
synchronize covers that wait. The wait is the GPU queue, not the
synchronize. The numbers below are that capture's.

tests/data/nsys-2024.6-loop-idle.sqlite is the same program captured the
same way on a V100 no other process was using. pressure starts about 2 us
after its launch returns, so the later iterations rewrite it.

tests/data/nsys-2024.6-graph.sqlite is tests/data/graph.cu, captured the
same way on a V100 no other process was using. nsys traced the graph as a
whole, its default with this driver, so the export has one graph row per
iteration and no kernel rows.
"""

import _path  # noqa: F401
import unittest

from stepceil.format import render_text
from stepceil.ledger import ASSUMPTION_EXCLUSIVE, account
from stepceil.nsys import list_ranges, load_path
from _path import ROOT

EXPORT = ROOT / "tests" / "data" / "nsys-2024.6-loop.sqlite"
IDLE_EXPORT = ROOT / "tests" / "data" / "nsys-2024.6-loop-idle.sqlite"
GRAPH_EXPORT = ROOT / "tests" / "data" / "nsys-2024.6-graph.sqlite"
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

    def test_the_wait_for_a_shared_gpu_is_the_queue(self):
        times = []
        queued = []
        for occurrence in range(2, 6):
            window = load_path(EXPORT, "step", occurrence)
            # The runtime call is wider than the synchronization record. It
            # is the synchronize, not a second, runtime piece around it.
            self.assertEqual([sync.name for sync in window.syncs], [SYNC])
            self.assertNotIn(SYNC, [call.name for call in window.runtime_ops])
            self.assertTrue(all(
                op.launch_end_ns is not None
                and op.launch_end_ns < op.start_ns
                for op in window.device_ops
            ))
            ledger = account(window)
            self.assertEqual(ledger["longest_blocker"]["kind"], "queue")
            self.assertEqual(ledger["longest_blocker"]["name"], "gpu queue")
            self.assertIsNone(ledger["contained"])
            self.assertEqual(ledger["separate"]["name"], "pressure")
            self.assertEqual(
                ledger["decision"], "do not rewrite a kernel from this report",
            )
            times.append(ledger["step_time_ns"])
            queued.append(ledger["queued_idle_ns"])
        self.assertEqual(times, [3999049, 2923571, 2934095, 2919721])
        self.assertEqual(queued[0], 3415498)
        self.assertTrue(all(
            wait * 10 > time * 7 for wait, time in zip(queued, times)
        ))

    def test_readme_shows_the_second_iteration(self):
        readme = (ROOT / "README.md").read_text()
        text = render_text(account(load_path(EXPORT, "step", 2)))
        excerpt = readme.split("--occurrence 2\n```\n\n```text\n")[2]
        excerpt = excerpt.split("```", 1)[0]
        self.assertIn(excerpt, text)
        self.assertIn("decision: do not rewrite a kernel from this report\n", text)


class RealIdleExportTest(unittest.TestCase):
    def test_without_a_queue_the_hot_kernel_is_the_rewrite(self):
        first = account(load_path(IDLE_EXPORT, "step", 1))
        self.assertEqual(first["decision"], "change gpu idle first")
        for occurrence in range(2, 6):
            ledger = account(load_path(IDLE_EXPORT, "step", occurrence))
            # The launch-to-start moment is microseconds, not a blocker.
            self.assertLess(ledger["queued_idle_ns"] * 100, ledger["step_time_ns"])
            self.assertEqual(ledger["longest_blocker"]["kind"], "kernel")
            self.assertEqual(ledger["longest_blocker"]["symbol"], "_Z8pressurePfii")
            self.assertTrue(ledger["allow_rewrite"])
            self.assertEqual(ledger["decision"], "rewrite pressure")
        second = account(load_path(IDLE_EXPORT, "step", 2))
        self.assertEqual(
            (second["step_time_ns"], second["ceiling_ns"], second["queued_idle_ns"]),
            (690991, 669563, 2067),
        )


class RealGraphExportTest(unittest.TestCase):
    def test_a_graph_traced_as_a_whole_is_the_busy_time(self):
        times = []
        for occurrence in range(1, 6):
            window = load_path(GRAPH_EXPORT, "step", occurrence)
            self.assertEqual(
                [(op.kind, op.name) for op in window.device_ops],
                [("graph", "cuda graph 1")],
            )
            self.assertIsNotNone(window.device_ops[0].launch_end_ns)
            ledger = account(window)
            # The step is the graph plus the launch and the synchronize around it.
            self.assertGreater(ledger["gpu_busy_ns"] * 10, ledger["step_time_ns"] * 9)
            self.assertEqual(ledger["longest_blocker"]["kind"], "graph")
            self.assertEqual(ledger["separate"]["name"], "cudaGraphLaunch_v10000")
            self.assertIsNone(ledger["hottest_kernel"])
            self.assertEqual(
                ledger["decision"], "do not rewrite a kernel from this report",
            )
            times.append((ledger["step_time_ns"], ledger["gpu_busy_ns"]))
        self.assertEqual(times[1], (693799, 678812))


if __name__ == "__main__":
    unittest.main()

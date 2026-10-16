import _path  # noqa: F401
import random
import unittest

from stepceil.errors import StepceilError
from stepceil.format import render_json, render_text
from stepceil.intervals import length, merge
from stepceil.ledger import (
    ASSUMPTION_EXCLUSIVE,
    ASSUMPTION_MULTI,
    ASSUMPTION_SAME,
    account,
    overlap_class,
)
from stepceil.model import DeviceOp, StepWindow, SyncOp
from _path import ROOT


def window(start, end, ops=(), syncs=(), name="step"):
    return StepWindow(name, start, end, tuple(ops), tuple(syncs))


def kernel(
    name, start, end, stream_id=1, device_id=0, context_id=1, process_id=None,
):
    return DeviceOp(
        "kernel", name, start, end, stream_id, device_id, context_id,
        process_id=process_id,
    )


def copy(start, end, name="memcpy HtoD", stream_id=1):
    return DeviceOp("memcpy", name, start, end, stream_id, 0, 1)


class LedgerTest(unittest.TestCase):
    def test_readme_sync_hides_the_hot_kernel(self):
        ledger = account(window(
            0, 1000,
            [kernel("pressure", 0, 80)],
            [SyncOp("context synchronize", 0, 1000)],
        ))
        self.assertEqual(ledger["gpu_busy_ns"], 80)
        self.assertEqual(ledger["gpu_idle_ns"], 920)
        self.assertEqual(ledger["longest_blocker"]["kind"], "sync")
        self.assertEqual(ledger["longest_blocker"]["start_ns"], 80)
        self.assertEqual(ledger["longest_blocker"]["blocking_ns"], 920)
        self.assertEqual(ledger["hottest_kernel"]["ceiling_ns"], 80)
        self.assertFalse(ledger["allow_rewrite"])
        self.assertIsNone(ledger["ceiling_refused"])
        self.assertIsNone(ledger["hottest_kernel"]["source"])
        readme = (ROOT / "README.md").read_text()
        self.assertIn(render_text(ledger), readme)
        self.assertIn(render_json(ledger), readme)

    def test_readme_overlap_uses_the_union(self):
        ledger = account(window(0, 100, [
            kernel("alpha", 0, 60, stream_id=1),
            kernel("beta", 40, 100, stream_id=2),
        ]))
        self.assertEqual(ledger["gpu_busy_ns"], 100)
        self.assertEqual(ledger["gpu_idle_ns"], 0)
        self.assertIsNone(ledger["ceiling_ns"])
        self.assertEqual(ledger["ceiling_refused"], ASSUMPTION_MULTI)
        self.assertEqual(ledger["overlap"]["overlap_ns"], 20)
        self.assertFalse(ledger["allow_rewrite"])
        readme = (ROOT / "README.md").read_text()
        self.assertIn(render_text(ledger), readme)

    def test_ceiling_is_the_sum_of_launches(self):
        ledger = account(window(0, 100, [
            kernel("pressure", 0, 6),
            kernel("pressure", 10, 16),
        ]))
        self.assertEqual(ledger["ceiling_ns"], 12)
        self.assertEqual(ledger["hottest_kernel"]["instances"], 2)
        self.assertTrue(ledger["allow_rewrite"])
        self.assertEqual(ledger["longest_blocker"]["kind"], "gap")
        self.assertEqual(ledger["longest_blocker"]["blocking_ns"], 84)
        self.assertEqual(
            ledger["gpu_busy_ns"] + ledger["gpu_idle_ns"],
            ledger["step_time_ns"],
        )

    def test_rewrite_threshold_boundaries(self):
        under = account(window(0, 100, [kernel("pressure", 0, 9)]))
        exact = account(window(0, 100, [kernel("pressure", 0, 10)]))
        self.assertFalse(under["allow_rewrite"])
        self.assertEqual(under["ceiling_ns"], 9)
        self.assertTrue(exact["allow_rewrite"])
        self.assertEqual(exact["ceiling_ns"], 10)

    def test_touching_launches_are_not_overlap(self):
        ledger = account(window(0, 30, [
            kernel("pressure", 0, 10),
            kernel("pressure", 10, 20),
        ]))
        self.assertEqual(ledger["assumption"], ASSUMPTION_EXCLUSIVE)
        self.assertEqual(ledger["ceiling_ns"], 20)
        self.assertEqual(ledger["longest_blocker"]["blocking_ns"], 10)
        self.assertEqual(ledger["gpu_busy_ns"], 20)

    def test_same_stream_overlap_withholds_the_ceiling(self):
        ledger = account(window(0, 20, [
            kernel("pressure", 0, 10),
            kernel("pressure", 5, 15),
        ]))
        self.assertEqual(ledger["ceiling_refused"], ASSUMPTION_SAME)
        self.assertEqual(ledger["gpu_busy_ns"], 15)
        self.assertIsNone(ledger["longest_blocker"])
        self.assertEqual(ledger["overlap"]["first"]["start_ns"], 0)
        self.assertEqual(ledger["overlap"]["first"]["end_ns"], 10)
        self.assertEqual(ledger["overlap"]["second"]["start_ns"], 5)
        self.assertEqual(ledger["overlap"]["second"]["end_ns"], 15)
        self.assertEqual(ledger["overlap"]["first"]["stream_id"], 1)
        self.assertEqual(ledger["overlap"]["second"]["stream_id"], 1)

    def test_different_contexts_count_as_overlap(self):
        ledger = account(window(0, 20, [
            kernel("pressure", 0, 10, context_id=1),
            kernel("other", 5, 15, context_id=2),
        ]))
        self.assertEqual(ledger["ceiling_refused"], ASSUMPTION_MULTI)
        self.assertEqual(ledger["overlap"]["first"]["context_id"], 1)
        self.assertEqual(ledger["overlap"]["second"]["context_id"], 2)
        self.assertEqual(ledger["overlap"]["first"]["stream_id"], 1)
        self.assertEqual(ledger["overlap"]["second"]["stream_id"], 1)

    def test_overlap_outside_the_step_does_not_count(self):
        ledger = account(window(0, 40, [
            kernel("pressure", 0, 100, stream_id=1),
            kernel("other", 50, 150, stream_id=2),
        ]))
        self.assertEqual(ledger["assumption"], ASSUMPTION_EXCLUSIVE)
        self.assertEqual(ledger["ceiling_ns"], 40)
        self.assertEqual(ledger["gpu_busy_ns"], 40)

    def test_other_gpu_outside_the_step_is_ignored(self):
        ledger = account(window(0, 100, [
            kernel("pressure", 0, 40, device_id=0),
            kernel("other", 200, 300, device_id=1),
        ]))
        self.assertEqual(ledger["ceiling_ns"], 40)

    def test_two_processes_inside_the_step_refuse(self):
        with self.assertRaises(StepceilError) as raised:
            account(window(0, 40, [
                kernel("pressure", 0, 10, process_id=10),
                kernel("other", 10, 20, process_id=11),
            ]))
        self.assertIn("process ids 10, 11", str(raised.exception))

        outside = account(window(0, 40, [
            kernel("pressure", 0, 40, process_id=10),
            kernel("other", 50, 90, process_id=11),
        ]))
        self.assertEqual(outside["hottest_kernel"]["name"], "pressure")
        self.assertEqual(outside["ceiling_ns"], 40)

        kept = account(window(0, 40, [
            kernel("pressure", 0, 20, process_id=10),
            kernel("tail", 20, 30, process_id=10),
            kernel("anon", 30, 40, process_id=None),
        ]))
        self.assertEqual(kept["ceiling_ns"], 20)
        self.assertEqual(kept["gpu_busy_ns"], 40)
        self.assertIsNone(kept["ceiling_refused"])

        with self.assertRaises(StepceilError) as raised_sync:
            account(window(
                0, 40,
                [kernel("pressure", 0, 10, process_id=10)],
                [SyncOp("cudaDeviceSynchronize", 0, 40, process_id=11)],
            ))
        self.assertIn("process ids 10, 11", str(raised_sync.exception))

        quiet = account(window(
            0, 40,
            [kernel("pressure", 0, 40, process_id=10)],
            [SyncOp("cudaDeviceSynchronize", 50, 90, process_id=11)],
        ))
        self.assertEqual(quiet["decision"], "rewrite pressure")

    def test_two_gpus_inside_the_step_refuse(self):
        with self.assertRaisesRegex(Exception, "device ids 0, 1"):
            account(window(0, 100, [
                kernel("pressure", 0, 40, device_id=0),
                kernel("other", 50, 80, device_id=1),
            ]))

    def test_partial_launches_are_clipped(self):
        ledger = account(window(100, 200, [
            kernel("pressure", 50, 150),
            kernel("pressure", 180, 250),
        ]))
        self.assertEqual(ledger["ceiling_ns"], 70)
        self.assertEqual(ledger["gpu_busy_ns"], 70)
        self.assertEqual(ledger["gpu_idle_ns"], 30)
        self.assertEqual(ledger["longest_blocker"]["name"], "pressure")
        self.assertEqual(ledger["longest_blocker"]["blocking_ns"], 50)

    def test_clipped_transfer_drops_its_byte_count(self):
        inside = DeviceOp(
            "memcpy", "memcpy HtoD", 0, 40, 1, 0, 1, (), 4096
        )
        kept = account(window(0, 40, [inside]))
        self.assertEqual(kept["longest_blocker"]["kind"], "memcpy")
        self.assertEqual(kept["longest_blocker"]["blocking_ns"], 40)
        self.assertEqual(kept["longest_blocker"]["bytes"], 4096)

        crossing = DeviceOp(
            "memcpy", "memcpy HtoD", 0, 80, 1, 0, 1, (), 4096
        )
        dropped = account(window(0, 40, [crossing]))
        self.assertEqual(dropped["longest_blocker"]["kind"], "memcpy")
        self.assertEqual(dropped["longest_blocker"]["blocking_ns"], 40)
        self.assertIsNone(dropped["longest_blocker"]["bytes"])

    def test_empty_step_is_one_gap(self):
        ledger = account(window(0, 100))
        self.assertEqual(ledger["gpu_busy_ns"], 0)
        self.assertEqual(ledger["ceiling_ns"], 0)
        self.assertIsNone(ledger["hottest_kernel"])
        self.assertEqual(ledger["longest_blocker"]["kind"], "gap")
        self.assertEqual(ledger["longest_blocker"]["blocking_ns"], 100)
        self.assertFalse(ledger["allow_rewrite"])

    def test_sync_does_not_take_kernel_time(self):
        ledger = account(window(
            0, 100,
            [kernel("pressure", 0, 50)],
            [SyncOp("context synchronize", 90, 100)],
        ))
        self.assertEqual(ledger["longest_blocker"]["kind"], "kernel")
        self.assertEqual(ledger["longest_blocker"]["blocking_ns"], 50)
        kinds = {
            ledger["longest_blocker"]["kind"],
        }
        self.assertIn("kernel", kinds)
        gap = account(window(
            0, 100,
            [kernel("small", 0, 10)],
            [SyncOp("context synchronize", 90, 100)],
        ))
        self.assertEqual(gap["longest_blocker"]["kind"], "gap")
        self.assertEqual(gap["longest_blocker"]["start_ns"], 10)
        self.assertEqual(gap["longest_blocker"]["end_ns"], 90)

    def test_adjacent_sync_coverage_merges_one_call(self):
        ledger = account(window(0, 100, syncs=[
            SyncOp("context synchronize", 0, 80),
            SyncOp("stream synchronize", 70, 100),
        ]))
        self.assertEqual(ledger["longest_blocker"]["name"], "context synchronize")
        self.assertEqual(ledger["longest_blocker"]["blocking_ns"], 80)

    def test_shorter_sync_does_not_shorten_a_longer_one(self):
        forward = [
            SyncOp("a", 0, 30),
            SyncOp("b", 0, 100),
        ]
        backward = list(reversed(forward))
        ledgers = [
            account(window(0, 100, syncs=forward)),
            account(window(0, 100, syncs=backward)),
        ]
        for ledger in ledgers:
            self.assertEqual(ledger["longest_blocker"]["name"], "b")
            self.assertEqual(ledger["longest_blocker"]["blocking_ns"], 100)
            self.assertEqual(ledger["longest_blocker"]["start_ns"], 0)
            self.assertEqual(ledger["longest_blocker"]["end_ns"], 100)
        split = account(window(
            0, 100,
            [kernel("pressure", 40, 60)],
            [SyncOp("context synchronize", 0, 100)],
        ))
        self.assertEqual(split["longest_blocker"]["kind"], "sync")
        self.assertEqual(split["longest_blocker"]["blocking_ns"], 40)
        self.assertEqual(split["longest_blocker"]["start_ns"], 0)
        self.assertEqual(split["longest_blocker"]["end_ns"], 40)

    def test_hottest_tie_uses_the_earlier_launch(self):
        ledger = account(window(0, 100, [
            kernel("alpha", 50, 90),
            kernel("beta", 0, 40),
        ]))
        self.assertEqual(ledger["hottest_kernel"]["name"], "beta")
        self.assertEqual(ledger["ceiling_ns"], 40)

    def test_longest_tie_uses_the_earlier_interval(self):
        ledger = account(window(0, 100, [
            kernel("pressure", 50, 100),
            copy(0, 50),
        ]))
        self.assertEqual(ledger["longest_blocker"]["kind"], "memcpy")
        self.assertEqual(ledger["longest_blocker"]["blocking_ns"], 50)

    def test_decision_is_not_the_allow_rewrite_flag(self):
        summed = account(window(0, 100, [
            kernel("pressure", 0, 6),
            kernel("pressure", 10, 16),
        ]))
        self.assertTrue(summed["allow_rewrite"])
        self.assertEqual(summed["longest_blocker"]["kind"], "gap")
        self.assertEqual(summed["decision"], "change gpu idle first")

        whole = account(window(0, 100, [kernel("pressure", 0, 100)]))
        self.assertEqual(whole["decision"], "rewrite pressure")

        pieces = [kernel("pressure", 0, 9)]
        cursor = 9
        for index, width in enumerate((8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 3)):
            pieces.append(kernel(f"other{index}", cursor, cursor + width))
            cursor += width
        self.assertEqual(cursor, 100)
        under = account(window(0, 100, pieces))
        self.assertFalse(under["allow_rewrite"])
        self.assertEqual(under["longest_blocker"]["name"], "pressure")
        self.assertEqual(under["decision"], "change pressure first")

        overlap = account(window(0, 100, [
            kernel("alpha", 0, 60, stream_id=1),
            kernel("beta", 40, 100, stream_id=2),
        ]))
        self.assertEqual(
            overlap["decision"], "do not rewrite a kernel from this report"
        )
        self.assertEqual(overlap["overlap"]["first"]["name"], "alpha")
        self.assertEqual(overlap["overlap"]["second"]["name"], "beta")
        self.assertIsNone(account(window(0, 30, [
            kernel("pressure", 0, 10),
            kernel("pressure", 10, 20),
        ]))["overlap"])

        crossed = account(window(0, 100, [
            kernel("wide", 0, 100, stream_id=1),
            kernel("nested", 10, 20, stream_id=1),
            kernel("other", 15, 25, stream_id=2),
        ]))
        self.assertEqual(crossed["overlap"]["first"]["name"], "wide")
        self.assertEqual(crossed["overlap"]["second"]["name"], "other")
        self.assertNotEqual(
            crossed["overlap"]["first"]["stream_id"],
            crossed["overlap"]["second"]["stream_id"],
        )

        clipped = account(window(0, 50, [
            kernel("alpha", -10, 40, stream_id=1),
            kernel("beta", 30, 80, stream_id=2),
        ]))
        self.assertEqual(clipped["overlap"]["first"]["start_ns"], 0)
        self.assertEqual(clipped["overlap"]["first"]["end_ns"], 40)
        self.assertEqual(clipped["overlap"]["second"]["start_ns"], 30)
        self.assertEqual(clipped["overlap"]["second"]["end_ns"], 50)

        longest = account(window(0, 100, [
            kernel("wide", 0, 100, stream_id=1),
            kernel("early", 0, 10, stream_id=2),
            kernel("long", 20, 90, stream_id=3),
        ]))
        self.assertEqual(longest["overlap"]["first"]["name"], "wide")
        self.assertEqual(longest["overlap"]["second"]["name"], "long")

        same_longer = account(window(0, 100, [
            kernel("wide", 0, 100, stream_id=1),
            kernel("nested", 0, 80, stream_id=1),
            kernel("other", 90, 100, stream_id=2),
        ]))
        self.assertEqual(same_longer["ceiling_refused"], ASSUMPTION_MULTI)
        self.assertEqual(same_longer["overlap"]["second"]["name"], "other")

    def test_overlap_pair_does_not_follow_row_order(self):
        named = [
            kernel("b", 0, 50, stream_id=1),
            kernel("a", 0, 50, stream_id=1),
            kernel("other", 10, 20, stream_id=2),
        ]
        contexts = [
            kernel("pressure", 0, 10, stream_id=7, context_id=2),
            kernel("pressure", 0, 10, stream_id=7, context_id=1),
        ]
        for ops, first in ((named, "a"), (list(reversed(named)), "a")):
            ledger = account(window(0, 50, ops))
            self.assertEqual(ledger["overlap"]["first"]["name"], first)
            self.assertEqual(ledger["overlap"]["second"]["name"], "other")
            self.assertEqual(ledger["overlap"]["overlap_ns"], 10)
        forward = account(window(0, 10, contexts))
        backward = account(window(0, 10, list(reversed(contexts))))
        self.assertEqual(forward["overlap"], backward["overlap"])
        self.assertEqual(forward["overlap"]["first"]["context_id"], 1)
        self.assertEqual(forward["overlap"]["second"]["context_id"], 2)
        self.assertEqual(forward["overlap"]["overlap_ns"], 10)

    def test_overlap_class_matches_pairs(self):
        rng = random.Random(1)
        for _ in range(40):
            ops = []
            for _index in range(rng.randint(0, 12)):
                start = rng.randint(0, 900)
                ops.append(kernel(
                    rng.choice(["a", "b"]),
                    start,
                    start + rng.randint(1, 100),
                    stream_id=rng.choice([1, 2]),
                    context_id=rng.choice([1, 2]),
                ))
            self.assertEqual(overlap_class(ops), _reference_overlap(ops))
            ledger = account(window(0, 1000, ops))
            self.assertEqual(
                ledger["gpu_busy_ns"] + ledger["gpu_idle_ns"],
                1000,
            )
            self.assertEqual(
                ledger["gpu_busy_ns"],
                length(merge((op.start_ns, op.end_ns) for op in ops)),
            )
            kind = _reference_overlap(ops)
            if kind is None:
                self.assertIsNone(ledger["ceiling_refused"])
                self.assertIsNone(ledger["overlap"])
            else:
                self.assertIsNone(ledger["ceiling_ns"])
                pair = ledger["overlap"]
                self.assertLess(pair["first"]["start_ns"], pair["second"]["end_ns"])
                self.assertLess(pair["second"]["start_ns"], pair["first"]["end_ns"])
                left = (pair["first"]["context_id"], pair["first"]["stream_id"])
                right = (pair["second"]["context_id"], pair["second"]["stream_id"])
                if kind == "multi":
                    self.assertNotEqual(left, right)
                else:
                    self.assertEqual(left, right)
                overlap_ns = (
                    min(pair["first"]["end_ns"], pair["second"]["end_ns"])
                    - max(pair["first"]["start_ns"], pair["second"]["start_ns"])
                )
                self.assertEqual(overlap_ns, _max_overlap_ns(ops, kind))
                self.assertEqual(pair["overlap_ns"], overlap_ns)


def _max_overlap_ns(ops, kind):
    best = 0
    for index, left in enumerate(ops):
        for right in ops[index + 1:]:
            if left.start_ns >= right.end_ns or right.start_ns >= left.end_ns:
                continue
            left_key = (left.context_id, left.stream_id)
            right_key = (right.context_id, right.stream_id)
            crossed = left_key != right_key
            if kind == "multi" and not crossed:
                continue
            if kind == "same" and crossed:
                continue
            overlap_ns = (
                min(left.end_ns, right.end_ns)
                - max(left.start_ns, right.start_ns)
            )
            if overlap_ns > best:
                best = overlap_ns
    return best


def _reference_overlap(ops):
    multi = False
    same = False
    for index, left in enumerate(ops):
        for right in ops[index + 1:]:
            if left.start_ns < right.end_ns and right.start_ns < left.end_ns:
                left_key = (left.context_id, left.stream_id)
                right_key = (right.context_id, right.stream_id)
                if left_key != right_key:
                    multi = True
                else:
                    same = True
    if multi:
        return "multi"
    if same:
        return "same"
    return None


if __name__ == "__main__":
    unittest.main()

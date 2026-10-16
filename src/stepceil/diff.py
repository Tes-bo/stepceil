"""Compare two ledgers of the same NVTX range.

faster is true only when the later step is strictly shorter. A withheld
ceiling does not block that comparison: both ledgers still have a step time.
"""

from stepceil.errors import StepceilError


def compare_ledgers(before, after):
    if before["range"] != after["range"]:
        raise StepceilError(
            f"range names differ: {before['range']!r} and {after['range']!r}"
        )
    before_time = before["step_time_ns"]
    after_time = after["step_time_ns"]
    return {
        "range": before["range"],
        "before_step_time_ns": before_time,
        "after_step_time_ns": after_time,
        "step_time_ns_delta": after_time - before_time,
        "faster": after_time < before_time,
        "before_longest_blocker": before["longest_blocker"],
        "after_longest_blocker": after["longest_blocker"],
        "before_ceiling_ns": before["ceiling_ns"],
        "after_ceiling_ns": after["ceiling_ns"],
        "before_allow_rewrite": before["allow_rewrite"],
        "after_allow_rewrite": after["allow_rewrite"],
        "before_ceiling_refused": before["ceiling_refused"],
        "after_ceiling_refused": after["ceiling_refused"],
        "before_decision": before["decision"],
        "after_decision": after["decision"],
    }

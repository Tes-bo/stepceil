"""Text and JSON renderings of one ledger. Both come from the same dict."""

import json


def render_json(ledger):
    return json.dumps(ledger, indent=2, ensure_ascii=False) + "\n"


def render_text(ledger):
    lines = [
        f"range: {ledger['range']}",
        f"step_start_ns: {ledger['step_start_ns']}",
        f"step_end_ns: {ledger['step_end_ns']}",
        f"step_time_ns: {ledger['step_time_ns']}",
        f"gpu_busy_ns: {ledger['gpu_busy_ns']}",
        f"gpu_idle_ns: {ledger['gpu_idle_ns']}",
        f"assumption: {ledger['assumption']}",
    ]
    lines.extend(_object_lines(
        "longest_blocker",
        ledger["longest_blocker"],
        ("kind", "name", "start_ns", "end_ns", "blocking_ns", "source"),
    ))
    lines.extend(_object_lines(
        "hottest_kernel",
        ledger["hottest_kernel"],
        ("name", "ceiling_ns", "instances", "source"),
    ))
    lines.append(f"ceiling_ns: {_cell(ledger['ceiling_ns'])}")
    lines.append(f"allow_rewrite: {_cell(ledger['allow_rewrite'])}")
    lines.append(f"ceiling_refused: {_cell(ledger['ceiling_refused'])}")
    lines.append(
        "rewrite_threshold_numerator: "
        f"{ledger['rewrite_threshold_numerator']}"
    )
    lines.append(
        "rewrite_threshold_denominator: "
        f"{ledger['rewrite_threshold_denominator']}"
    )
    return "\n".join(lines) + "\n"


def _object_lines(prefix, obj, fields):
    lines = []
    for field in fields:
        value = None if obj is None else obj[field]
        lines.append(f"{prefix}.{field}: {_cell(value)}")
    return lines


def _cell(value):
    if value is None:
        return "none"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)

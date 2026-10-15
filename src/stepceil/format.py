"""Text and JSON renderings of one ledger. Both come from the same dict."""

import json


def render_ranges_text(report):
    lines = [f"ranges: {len(report['ranges'])}"]
    for item in report["ranges"]:
        lines.append(f"name: {item['name']}")
        lines.append(f"count: {item['count']}")
        lines.append(f"closed: {item['closed']}")
    return "\n".join(lines) + "\n"


def render_json(ledger):
    return json.dumps(ledger, indent=2, ensure_ascii=False) + "\n"


def render_diff_text(report):
    lines = [
        f"range: {report['range']}",
        f"before_step_time_ns: {report['before_step_time_ns']}",
        f"after_step_time_ns: {report['after_step_time_ns']}",
        f"step_time_ns_delta: {report['step_time_ns_delta']}",
        f"faster: {_cell(report['faster'])}",
    ]
    lines.extend(_object_lines(
        "before_longest_blocker",
        report["before_longest_blocker"],
        ("kind", "name", "start_ns", "end_ns", "blocking_ns", "source"),
    ))
    lines.extend(_object_lines(
        "after_longest_blocker",
        report["after_longest_blocker"],
        ("kind", "name", "start_ns", "end_ns", "blocking_ns", "source"),
    ))
    lines.append(f"before_ceiling_ns: {_cell(report['before_ceiling_ns'])}")
    lines.append(f"after_ceiling_ns: {_cell(report['after_ceiling_ns'])}")
    lines.append(f"before_allow_rewrite: {_cell(report['before_allow_rewrite'])}")
    lines.append(f"after_allow_rewrite: {_cell(report['after_allow_rewrite'])}")
    lines.append(
        f"before_ceiling_refused: {_cell(report['before_ceiling_refused'])}"
    )
    lines.append(
        f"after_ceiling_refused: {_cell(report['after_ceiling_refused'])}"
    )
    return "\n".join(lines) + "\n"


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

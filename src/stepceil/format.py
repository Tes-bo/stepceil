"""Text and JSON renderings of one ledger. Both come from the same dict."""

import json


def render_ranges_text(report):
    lines = [f"ranges: {len(report['ranges'])}"]
    for item in report["ranges"]:
        lines.append(f"name: {item['name']}")
        lines.append(f"count: {item['count']}")
        lines.append(f"closed: {item['closed']}")
        lines.append(f"shortest_ns: {_cell(item['shortest_ns'])}")
        lines.append(f"longest_ns: {_cell(item['longest_ns'])}")
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
        ("kind", "name", "symbol", "start_ns", "end_ns", "blocking_ns", "source", "bytes"),
    ))
    lines.extend(_object_lines(
        "after_longest_blocker",
        report["after_longest_blocker"],
        ("kind", "name", "symbol", "start_ns", "end_ns", "blocking_ns", "source", "bytes"),
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
    lines.append(f"before_decision: {report['before_decision']}")
    lines.append(f"after_decision: {report['after_decision']}")
    return "\n".join(lines) + "\n"


def _contained_lines(prefix, obj):
    lines = _object_lines(prefix, obj, ("kind", "name", "symbol", "blocking_ns"))
    nested = None if obj is None else obj["contained"]
    lines.extend(_object_lines(
        f"{prefix}.contained",
        nested,
        ("kind", "name", "symbol", "blocking_ns"),
    ))
    return lines


def _further_lines(further):
    if further is None:
        return ["further: none"]
    lines = [f"further: {len(further)}"]
    fields = (
        "kind", "name", "symbol", "start_ns", "end_ns",
        "blocking_ns", "source", "bytes",
    )
    for index, item in enumerate(further):
        prefix = f"further.{index}"
        lines.extend(_object_lines(prefix, item, fields))
        lines.extend(_contained_lines(f"{prefix}.contained", item["contained"]))
    return lines


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
        ("kind", "name", "symbol", "start_ns", "end_ns", "blocking_ns", "source", "bytes"),
    ))
    lines.extend(_contained_lines("contained", ledger["contained"]))
    lines.extend(_object_lines(
        "separate",
        ledger["separate"],
        ("kind", "name", "symbol", "start_ns", "end_ns", "blocking_ns", "source", "bytes"),
    ))
    separate_contained = None
    if ledger["separate"] is not None:
        separate_contained = ledger["separate"]["contained"]
    lines.extend(_contained_lines("separate.contained", separate_contained))
    lines.extend(_further_lines(ledger["further"]))
    lines.extend(_object_lines(
        "hottest_kernel",
        ledger["hottest_kernel"],
        ("name", "symbol", "ceiling_ns", "instances", "source"),
    ))
    lines.append(f"ceiling_ns: {_cell(ledger['ceiling_ns'])}")
    lines.append(f"allow_rewrite: {_cell(ledger['allow_rewrite'])}")
    lines.append(f"ceiling_refused: {_cell(ledger['ceiling_refused'])}")
    lines.extend(_overlap_lines(ledger["overlap"]))
    lines.append(
        "rewrite_threshold_numerator: "
        f"{ledger['rewrite_threshold_numerator']}"
    )
    lines.append(
        "rewrite_threshold_denominator: "
        f"{ledger['rewrite_threshold_denominator']}"
    )
    lines.append(f"decision: {ledger['decision']}")
    return "\n".join(lines) + "\n"


def _overlap_lines(overlap):
    overlap_ns = None if overlap is None else overlap["overlap_ns"]
    lines = [f"overlap.overlap_ns: {_cell(overlap_ns)}"]
    for side in ("first", "second"):
        obj = None if overlap is None else overlap[side]
        lines.extend(_object_lines(
            f"overlap.{side}",
            obj,
            ("kind", "name", "start_ns", "end_ns", "stream_id", "context_id", "bytes"),
        ))
    return lines


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

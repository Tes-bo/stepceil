"""Minimal MCP stdio server for the stepceil ledger.

step_ceiling, step_diff, and step_ranges call the same functions as the
command line. They do not compute a second ceiling.

The stdio transport is one JSON-RPC message per line, UTF-8, with no
embedded newline. It is not the Content-Length framing of LSP.
"""

import json
import sys
from pathlib import Path

from stepceil import VERSION
from stepceil.errors import StepceilError
from stepceil.diff import compare_ledgers
from stepceil.format import render_json
from stepceil.ledger import account
from stepceil.nsys import list_ranges, load_path

# Oldest first. A client that asks for another version gets the latest.
SUPPORTED_PROTOCOLS = (
    "2024-11-05",
    "2025-03-26",
    "2025-06-18",
    "2025-11-25",
)

TOOL = {
    "name": "step_ceiling",
    "description": (
        "Read one NVTX range from an nsys sqlite export and return the "
        "speedup ceiling of that GPU step. Do not estimate the ceiling "
        "without this tool. Obey decision. Rewrite a kernel only when "
        "decision is rewrite followed by that kernel's name. When overlap "
        "is present, overlap_ns is the length of that intersection. Do not add "
        "their durations or their byte counts. bytes is null unless that "
        "side is a copy or memset that lies entirely inside the step. "
        "A runtime blocker is a host API call, and an nvtx blocker is another "
        "range inside the step. Neither is GPU work. When several calls cover "
        "the same idle, the narrowest one is the blocker. contained is the "
        "longest operation strictly inside that piece. It is not a second "
        "decision. contained.contained opens that result once more, then "
        "stops. separate is the longest piece that does not meet the blocker. "
        "It is not a second decision. further lists each later piece that "
        "lasts at least a tenth of the step and does not meet a piece already "
        "listed. It is not a decision. repeated lists a host API, and an "
        "NVTX range whose every idle piece is shorter than a tenth of the "
        "step, when two or more calls together last at least that tenth. "
        "A range inside another listed range is left off. Do not add that "
        "idle to a blocker, and do not add two repeated entries together. "
        "short_idle_ns is idle those pieces do not cover. "
        "short_idle_longest_ns is the longest stretch of that idle, not a "
        "call and not a decision. Do not add it to a blocker or to repeated. "
        "More than one process in the step is an error."
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to an nsys sqlite export.",
            },
            "range": {
                "type": "string",
                "description": "NVTX range name that marks the one step.",
            },
        },
        "required": ["path", "range"],
        "additionalProperties": False,
    },
}

RANGES_TOOL = {
    "name": "step_ranges",
    "description": (
        "List NVTX range names in an nsys sqlite export. Account a name "
        "only when count is 1 and closed is 1. shortest_ns and longest_ns "
        "are the closed occurrences. They do not choose a window. Do not "
        "guess a range name or replace a missing range with a start and "
        "end time."
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to an nsys sqlite export.",
            },
        },
        "required": ["path"],
        "additionalProperties": False,
    },
}

DIFF_TOOL = {
    "name": "step_diff",
    "description": (
        "Compare two nsys sqlite exports of the same NVTX range. "
        "faster is true only when the later step is strictly shorter. "
        "Do not claim a speedup unless faster is true."
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "before": {
                "type": "string",
                "description": "Path to the earlier nsys sqlite export.",
            },
            "after": {
                "type": "string",
                "description": "Path to the later nsys sqlite export.",
            },
            "range": {
                "type": "string",
                "description": "NVTX range name that marks the one step.",
            },
        },
        "required": ["before", "after", "range"],
        "additionalProperties": False,
    },
}


def serve(stdin=None, stdout=None):
    """Answer one message per line until stdin closes. Return 0."""
    incoming = sys.stdin.buffer if stdin is None else stdin
    outgoing = sys.stdout.buffer if stdout is None else stdout
    while True:
        line = incoming.readline()
        if not line:
            return 0
        if not line.strip():
            continue
        try:
            message = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            _write_message(outgoing, _error(None, -32700, "parse error"))
            continue
        response = _handle(message)
        if response is not None:
            _write_message(outgoing, response)


def _handle(message):
    if not isinstance(message, dict):
        return _error(None, -32600, "request is not an object")
    method = message.get("method")
    if not isinstance(method, str):
        # A response to a request this server never sends.
        return None
    if "id" not in message:
        # A notification gets no reply, known or not.
        return None
    msg_id = message["id"]
    params = message.get("params") or {}
    if not isinstance(params, dict):
        return _error(msg_id, -32602, "params is not an object")
    if method == "initialize":
        protocol = params.get("protocolVersion")
        if protocol not in SUPPORTED_PROTOCOLS:
            protocol = SUPPORTED_PROTOCOLS[-1]
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": {
                "protocolVersion": protocol,
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "stepceil", "version": VERSION},
            },
        }
    if method == "ping":
        return {"jsonrpc": "2.0", "id": msg_id, "result": {}}
    if method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": {"tools": [TOOL, DIFF_TOOL, RANGES_TOOL]},
        }
    if method == "tools/call":
        return _call(msg_id, params)
    return _error(msg_id, -32601, f"unknown method: {method}")


def _call(msg_id, params):
    name = params.get("name")
    arguments = params.get("arguments") or {}
    if not isinstance(arguments, dict):
        return _tool_result(msg_id, "arguments must be an object", True)
    if name == TOOL["name"]:
        return _ceiling_call(msg_id, arguments)
    if name == DIFF_TOOL["name"]:
        return _diff_call(msg_id, arguments)
    if name == RANGES_TOOL["name"]:
        return _ranges_call(msg_id, arguments)
    return _error(msg_id, -32602, f"unknown tool: {name}")


def _ceiling_call(msg_id, arguments):
    path = arguments.get("path")
    range_name = arguments.get("range")
    if not isinstance(path, str) or not isinstance(range_name, str):
        return _tool_result(msg_id, "path and range must be strings", True)
    try:
        ledger = account(load_path(Path(path), range_name))
    except StepceilError as exc:
        return _tool_result(msg_id, str(exc), True)
    return _tool_result(msg_id, render_json(ledger), False)


def _ranges_call(msg_id, arguments):
    path = arguments.get("path")
    if not isinstance(path, str):
        return _tool_result(msg_id, "path must be a string", True)
    try:
        report = list_ranges(Path(path))
    except StepceilError as exc:
        return _tool_result(msg_id, str(exc), True)
    return _tool_result(msg_id, render_json(report), False)


def _diff_call(msg_id, arguments):
    before = arguments.get("before")
    after = arguments.get("after")
    range_name = arguments.get("range")
    if (
        not isinstance(before, str)
        or not isinstance(after, str)
        or not isinstance(range_name, str)
    ):
        return _tool_result(
            msg_id, "before, after, and range must be strings", True
        )
    try:
        report = compare_ledgers(
            account(load_path(Path(before), range_name)),
            account(load_path(Path(after), range_name)),
        )
    except StepceilError as exc:
        return _tool_result(msg_id, str(exc), True)
    return _tool_result(msg_id, render_json(report), False)


def _tool_result(msg_id, text, is_error):
    return {
        "jsonrpc": "2.0",
        "id": msg_id,
        "result": {
            "content": [{"type": "text", "text": text}],
            "isError": is_error,
        },
    }


def _error(msg_id, code, message):
    return {
        "jsonrpc": "2.0",
        "id": msg_id,
        "error": {"code": code, "message": message},
    }


def _write_message(stream, payload):
    # json.dumps escapes newlines inside strings, so the line is one message.
    stream.write(json.dumps(payload).encode("utf-8") + b"\n")
    stream.flush()

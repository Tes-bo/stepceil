"""Minimal MCP stdio server for the stepceil ledger.

step_ceiling, step_diff, and step_ranges call the same functions as the
command line. They do not compute a second ceiling.
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

SUPPORTED_PROTOCOLS = (
    "2024-11-05",
    "2025-03-26",
    "2025-06-18",
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
        "A runtime blocker is a host API call, not GPU work. "
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
    incoming = sys.stdin.buffer if stdin is None else stdin
    outgoing = sys.stdout.buffer if stdout is None else stdout
    while True:
        try:
            message = _read_message(incoming)
        except (StepceilError, json.JSONDecodeError, UnicodeError, ValueError):
            return 1
        if message is None:
            return 0
        response = _handle(message)
        if response is not None:
            _write_message(outgoing, response)


def _handle(message):
    if not isinstance(message, dict):
        return _error(None, -32600, "request is not an object")
    method = message.get("method")
    msg_id = message.get("id")
    if isinstance(method, str) and method.startswith("notifications/"):
        return None
    if method == "initialize":
        params = message.get("params") or {}
        protocol = params.get("protocolVersion") or SUPPORTED_PROTOCOLS[0]
        if protocol not in SUPPORTED_PROTOCOLS:
            protocol = SUPPORTED_PROTOCOLS[0]
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
        return _call(msg_id, message.get("params") or {})
    return _error(msg_id, -32601, f"unknown method: {method}")


def _call(msg_id, params):
    name = params.get("name")
    arguments = params.get("arguments") or {}
    if name == TOOL["name"]:
        return _ceiling_call(msg_id, arguments)
    if name == DIFF_TOOL["name"]:
        return _diff_call(msg_id, arguments)
    if name == RANGES_TOOL["name"]:
        return _ranges_call(msg_id, arguments)
    return _tool_result(msg_id, f"unknown tool: {name}", True)


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


def _read_message(stream):
    header = b""
    while b"\r\n\r\n" not in header:
        chunk = stream.read(1)
        if not chunk:
            if header:
                raise StepceilError("incomplete MCP header")
            return None
        header += chunk
        if len(header) > 1024 * 1024:
            raise StepceilError("MCP header is too large")
    raw_header, rest = header.split(b"\r\n\r\n", 1)
    length = None
    for line in raw_header.split(b"\r\n"):
        if line.lower().startswith(b"content-length:"):
            length = int(line.split(b":", 1)[1].strip())
    if length is None or length < 0:
        raise StepceilError("MCP header has no Content-Length")
    body = rest
    while len(body) < length:
        chunk = stream.read(length - len(body))
        if not chunk:
            raise StepceilError("incomplete MCP body")
        body += chunk
    return json.loads(body.decode("utf-8"))


def _write_message(stream, payload):
    body = json.dumps(payload).encode("utf-8")
    stream.write(f"Content-Length: {len(body)}\r\n\r\n".encode("ascii"))
    stream.write(body)
    stream.flush()

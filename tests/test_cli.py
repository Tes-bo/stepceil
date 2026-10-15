import _path  # noqa: F401
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from stepceil import VERSION
from stepceil.format import render_json
from stepceil.ledger import account
from stepceil.mcp_server import serve
from stepceil.nsys import load_path
from _path import ROOT, SRC
from support import write_export


def run_cli(*args):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC)
    return subprocess.run(
        [sys.executable, "-m", "stepceil", *args],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
    )


def frame(payload):
    body = json.dumps(payload).encode()
    return f"Content-Length: {len(body)}\r\n\r\n".encode() + body


def frames(blob):
    parsed = []
    rest = blob
    while rest:
        head, rest = rest.split(b"\r\n\r\n", 1)
        length = int(head.split(b":", 1)[1].strip())
        parsed.append(json.loads(rest[:length].decode()))
        rest = rest[length:]
    return parsed


class CliTest(unittest.TestCase):
    def test_text_and_json_match_the_ledger(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "report.sqlite"
            write_export(
                path,
                [{"text": "step", "start": 0, "end": 1000}],
                kernels=[{"name": "pressure", "start": 0, "end": 80}],
                syncs=[{"sync_type": 4, "start": 0, "end": 1000}],
            )
            expected = render_json(account(load_path(path, "step")))
            text = run_cli(str(path), "--range", "step")
            as_json = run_cli(str(path), "--range", "step", "--json")
            self.assertEqual(text.returncode, 0)
            self.assertEqual(text.stderr, "")
            self.assertIn("allow_rewrite: false", text.stdout)
            self.assertEqual(as_json.returncode, 0)
            self.assertEqual(as_json.stdout, expected)

    def test_two_ranges_exit_2(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "report.sqlite"
            write_export(path, [
                {"text": "step", "start": 0, "end": 10},
                {"text": "step", "start": 20, "end": 30},
            ])
            result = run_cli(str(path), "--range", "step", "--json")
            self.assertEqual(result.returncode, 2)
            self.assertEqual(result.stdout, "")
            self.assertIn("found 2", result.stderr)
            self.assertTrue(result.stderr.startswith("stepceil:"))

    def test_overlap_still_exits_0(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "report.sqlite"
            write_export(
                path,
                [{"text": "step", "start": 0, "end": 100}],
                kernels=[
                    {"name": "alpha", "start": 0, "end": 60, "stream_id": 1},
                    {"name": "beta", "start": 40, "end": 100, "stream_id": 2},
                ],
            )
            result = run_cli(str(path), "--range", "step", "--json")
            self.assertEqual(result.returncode, 0)
            self.assertIsNone(json.loads(result.stdout)["ceiling_ns"])

    def test_version_and_mcp_arguments(self):
        version = run_cli("--version")
        self.assertEqual(version.returncode, 0)
        self.assertIn(VERSION, version.stdout)
        extra = run_cli("mcp", "--range", "step")
        self.assertEqual(extra.returncode, 2)
        self.assertIn("mcp takes no arguments", extra.stderr)

    def test_pyproject_version(self):
        text = (ROOT / "pyproject.toml").read_text()
        self.assertIn(f'version = "{VERSION}"', text)

    def test_mcp_returns_the_json_ledger(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "report.sqlite"
            write_export(
                path,
                [{"text": "step", "start": 0, "end": 40}],
                kernels=[{"name": "pressure", "start": 0, "end": 40}],
            )
            incoming = frame({
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {"protocolVersion": "2024-11-05"},
            })
            incoming += frame({
                "jsonrpc": "2.0",
                "method": "notifications/initialized",
            })
            incoming += frame({
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {
                    "name": "step_ceiling",
                    "arguments": {"path": str(path), "range": "step"},
                },
            })
            stdout = tempfile.TemporaryFile()
            try:
                code = serve(stdin=_Bytes(incoming), stdout=stdout)
                stdout.seek(0)
                messages = frames(stdout.read())
            finally:
                stdout.close()
            self.assertEqual(code, 0)
            self.assertEqual(messages[0]["result"]["serverInfo"]["name"], "stepceil")
            self.assertEqual(len(messages), 2)
            body = messages[1]["result"]["content"][0]["text"]
            self.assertFalse(messages[1]["result"]["isError"])
            self.assertEqual(json.loads(body)["ceiling_ns"], 40)

    def test_mcp_tool_error_on_a_bad_report(self):
        incoming = frame({
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "step_ceiling",
                "arguments": {"path": "/no/such.sqlite", "range": "step"},
            },
        })
        stdout = tempfile.TemporaryFile()
        try:
            serve(stdin=_Bytes(incoming), stdout=stdout)
            stdout.seek(0)
            message = frames(stdout.read())[0]
        finally:
            stdout.close()
        self.assertTrue(message["result"]["isError"])
        self.assertIn("no such file", message["result"]["content"][0]["text"])


class _Bytes:
    def __init__(self, blob):
        self._blob = blob
        self._at = 0

    def read(self, count):
        piece = self._blob[self._at:self._at + count]
        self._at += len(piece)
        return piece

    def write(self, blob):
        raise AssertionError("stdin is not written")

    def flush(self):
        return None


if __name__ == "__main__":
    unittest.main()

import _path  # noqa: F401
import io
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
    """One MCP stdio message: a JSON line."""
    return json.dumps(payload).encode() + b"\n"


def frames(blob):
    return [json.loads(line) for line in blob.splitlines() if line.strip()]


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
                code = serve(stdin=io.BytesIO(incoming), stdout=stdout)
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
            serve(stdin=io.BytesIO(incoming), stdout=stdout)
            stdout.seek(0)
            message = frames(stdout.read())[0]
        finally:
            stdout.close()
        self.assertTrue(message["result"]["isError"])
        self.assertIn("no such file", message["result"]["content"][0]["text"])

    def test_mcp_is_one_json_message_per_line(self):
        incoming = (
            b'{"jsonrpc":"2.0","id":1,"method":"initialize",'
            b'"params":{"protocolVersion":"2099-01-01"}}\n'
            b"\n"
            b"{not json\n"
            b'{"jsonrpc":"2.0","method":"notifications/unknown"}\n'
            b'{"jsonrpc":"2.0","id":2,"method":"tools/call",'
            b'"params":{"name":"nope","arguments":{}}}\n'
            b'{"jsonrpc":"2.0","id":3,"method":"tools/call",'
            b'"params":{"name":"step_ranges","arguments":[]}}\n'
            b'{"jsonrpc":"2.0","id":4,"method":"ping"}\n'
        )
        stdout = io.BytesIO()
        self.assertEqual(serve(stdin=io.BytesIO(incoming), stdout=stdout), 0)
        lines = stdout.getvalue().split(b"\n")
        self.assertEqual(lines[-1], b"")
        messages = [json.loads(line) for line in lines[:-1]]
        self.assertEqual([message.get("id") for message in messages], [1, None, 2, 3, 4])
        self.assertEqual(messages[0]["result"]["protocolVersion"], "2025-11-25")
        self.assertEqual(messages[1]["error"]["code"], -32700)
        self.assertEqual(messages[2]["error"]["code"], -32602)
        self.assertTrue(messages[3]["result"]["isError"])
        self.assertEqual(messages[4]["result"], {})

    def test_mcp_answers_over_a_pipe_before_stdin_closes(self):
        env = os.environ.copy()
        env["PYTHONPATH"] = str(SRC)
        process = subprocess.Popen(
            [sys.executable, "-m", "stepceil", "mcp"],
            cwd=ROOT,
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
        )
        try:
            process.stdin.write(frame({
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": "step_ceiling",
                    "arguments": {"path": "examples/sync.sqlite", "range": "step"},
                },
            }))
            process.stdin.flush()
            reply = json.loads(process.stdout.readline())
        finally:
            process.stdin.close()
            code = process.wait(timeout=10)
            process.stdout.close()
        self.assertEqual(code, 0)
        self.assertFalse(reply["result"]["isError"])
        ledger = json.loads(reply["result"]["content"][0]["text"])
        self.assertEqual(ledger["decision"], "change context synchronize first")

    def test_occurrence_on_the_command_line_and_over_mcp(self):
        loop = "examples/loop.sqlite"
        picked = run_cli(loop, "--range", "step", "--occurrence", "2", "--json")
        self.assertEqual(picked.returncode, 0, picked.stderr)
        ledger = json.loads(picked.stdout)
        self.assertEqual(ledger["occurrence"], 2)
        self.assertEqual(ledger["decision"], "rewrite pressure")

        refused = run_cli(loop, "--range", "step")
        self.assertEqual(refused.returncode, 2)
        self.assertIn("found 3; choose one with --occurrence N", refused.stderr)
        for bad in ("0", "x"):
            invalid = run_cli(loop, "--range", "step", "--occurrence", bad)
            self.assertEqual(invalid.returncode, 2)
            self.assertIn("at least 1", invalid.stderr)

        compared = run_cli(
            "diff", loop, loop, "--range", "step", "--occurrence", "3", "--json",
        )
        self.assertEqual(compared.returncode, 0, compared.stderr)
        self.assertEqual(json.loads(compared.stdout)["occurrence"], 3)

        def call(msg_id, name, arguments):
            return frame({
                "jsonrpc": "2.0",
                "id": msg_id,
                "method": "tools/call",
                "params": {"name": name, "arguments": arguments},
            })

        incoming = (
            call(1, "step_ceiling", {"path": loop, "range": "step", "occurrence": 2})
            + call(2, "step_ceiling", {"path": loop, "range": "step", "occurrence": 0})
            + call(3, "step_diff", {
                "before": loop, "after": loop, "range": "step", "occurrence": 1,
            })
        )
        stdout = io.BytesIO()
        cwd = os.getcwd()
        os.chdir(ROOT)
        try:
            serve(stdin=io.BytesIO(incoming), stdout=stdout)
        finally:
            os.chdir(cwd)
        replies = [message["result"] for message in frames(stdout.getvalue())]
        self.assertFalse(replies[0]["isError"])
        self.assertEqual(json.loads(replies[0]["content"][0]["text"]), ledger)
        self.assertTrue(replies[1]["isError"])
        self.assertIn("at least 1", replies[1]["content"][0]["text"])
        self.assertFalse(replies[2]["isError"])
        report = json.loads(replies[2]["content"][0]["text"])
        self.assertEqual(report["occurrence"], 1)
        self.assertEqual(report["before_decision"], "change cudaMalloc_v3020 first")

    def test_diff_is_faster_only_when_the_step_shrinks(self):
        with tempfile.TemporaryDirectory() as tmp:
            before = Path(tmp) / "before.sqlite"
            after = Path(tmp) / "after.sqlite"
            same = Path(tmp) / "same.sqlite"
            write_export(
                before,
                [{"text": "step", "start": 0, "end": 1000}],
                kernels=[{"name": "pressure", "start": 0, "end": 80}],
                syncs=[{"sync_type": 4, "start": 0, "end": 1000}],
            )
            write_export(
                after,
                [{"text": "step", "start": 0, "end": 200}],
                kernels=[{"name": "pressure", "start": 0, "end": 80}],
                syncs=[{"sync_type": 4, "start": 0, "end": 200}],
            )
            write_export(
                same,
                [{"text": "step", "start": 0, "end": 1000}],
                kernels=[{"name": "pressure", "start": 0, "end": 80}],
            )
            shorter = run_cli(
                "diff", str(before), str(after), "--range", "step", "--json"
            )
            self.assertEqual(shorter.returncode, 0, shorter.stderr)
            report = json.loads(shorter.stdout)
            self.assertEqual(report["step_time_ns_delta"], -800)
            self.assertTrue(report["faster"])
            self.assertEqual(report["before_longest_blocker"]["kind"], "sync")
            tied = run_cli(
                "diff", str(before), str(same), "--range", "step", "--json"
            )
            self.assertFalse(json.loads(tied.stdout)["faster"])
            text = run_cli("diff", str(before), str(after), "--range", "step")
            self.assertIn("faster: true", text.stdout)
            self.assertIn("step_time_ns_delta: -800", text.stdout)

    def test_ranges_lists_names_without_choosing_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "report.sqlite"
            write_export(path, [
                {"text": "step", "start": 0, "end": 10},
                {"text": "inner", "start": 0, "end": 4},
                {"text": "inner", "start": 4, "end": 8},
                {"text": "inner", "start": 8, "end": None},
                {"text": "flat", "start": 5, "end": 5},
                {"text": "step", "start": 0, "end": 10, "event_type": 34},
            ])
            result = run_cli("ranges", str(path), "--json")
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads(result.stdout)
            self.assertEqual(report["ranges"], [
                {
                    "name": "flat",
                    "count": 1,
                    "closed": 0,
                    "shortest_ns": None,
                    "longest_ns": None,
                },
                {
                    "name": "inner",
                    "count": 3,
                    "closed": 2,
                    "shortest_ns": 4,
                    "longest_ns": 4,
                },
                {
                    "name": "step",
                    "count": 1,
                    "closed": 1,
                    "shortest_ns": 10,
                    "longest_ns": 10,
                },
            ])
            text = run_cli("ranges", str(path))
            self.assertIn("ranges: 3", text.stdout)
            self.assertIn(
                "name: step\ncount: 1\nclosed: 1\nshortest_ns: 10\nlongest_ns: 10",
                text.stdout,
            )
            self.assertIn("shortest_ns: none", text.stdout)
            missing = run_cli("ranges", str(Path(tmp) / "missing.sqlite"))
            self.assertEqual(missing.returncode, 2)
            self.assertTrue(missing.stderr.startswith("stepceil:"))
            empty = Path(tmp) / "empty.sqlite"
            write_export(empty, [])
            listed = json.loads(run_cli("ranges", str(empty), "--json").stdout)
            self.assertEqual(listed["ranges"], [])
            incoming = frame({
                "jsonrpc": "2.0",
                "id": 4,
                "method": "tools/call",
                "params": {
                    "name": "step_ranges",
                    "arguments": {"path": str(path)},
                },
            })
            stdout = tempfile.TemporaryFile()
            try:
                serve(stdin=io.BytesIO(incoming), stdout=stdout)
                stdout.seek(0)
                message = frames(stdout.read())[0]
            finally:
                stdout.close()
            self.assertFalse(message["result"]["isError"])
            self.assertEqual(
                json.loads(message["result"]["content"][0]["text"]),
                report,
            )


if __name__ == "__main__":
    unittest.main()

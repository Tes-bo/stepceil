"""Command line. `stepceil mcp` serves the same ledger over MCP."""

import argparse
import sys
from pathlib import Path

from stepceil import VERSION
from stepceil.errors import StepceilError
from stepceil.diff import compare_ledgers
from stepceil.format import render_diff_text, render_json, render_text
from stepceil.ledger import account
from stepceil.nsys import load_path


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    if argv[:1] == ["mcp"]:
        if argv != ["mcp"]:
            sys.stderr.write("stepceil: mcp takes no arguments\n")
            return 2
        from stepceil.mcp_server import serve
        return serve()
    if argv[:1] == ["diff"]:
        return _diff(argv[1:])

    parser = argparse.ArgumentParser(
        prog="stepceil",
        description=(
            "Speedup ceiling of one GPU step from an nsys sqlite export."
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"stepceil {VERSION}",
    )
    parser.add_argument("sqlite", help="path to an nsys sqlite export")
    parser.add_argument(
        "--range",
        required=True,
        help="NVTX range name that marks the one step",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="print the ledger as JSON",
    )
    args = parser.parse_args(argv)
    try:
        ledger = account(load_path(Path(args.sqlite), args.range))
    except StepceilError as exc:
        sys.stderr.write(f"stepceil: {exc}\n")
        return 2
    text = render_json(ledger) if args.json else render_text(ledger)
    sys.stdout.write(text)
    return 0


def _diff(argv):
    parser = argparse.ArgumentParser(
        prog="stepceil diff",
        description="Compare two ledgers of the same NVTX range.",
    )
    parser.add_argument("before", help="nsys sqlite export from before the change")
    parser.add_argument("after", help="nsys sqlite export from after the change")
    parser.add_argument(
        "--range",
        required=True,
        help="NVTX range name that marks the one step",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="print the comparison as JSON",
    )
    args = parser.parse_args(argv)
    try:
        before = account(load_path(Path(args.before), args.range))
        after = account(load_path(Path(args.after), args.range))
        report = compare_ledgers(before, after)
    except StepceilError as exc:
        sys.stderr.write(f"stepceil: {exc}\n")
        return 2
    text = render_json(report) if args.json else render_diff_text(report)
    sys.stdout.write(text)
    return 0

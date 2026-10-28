# stepceil

[![MIT license](https://img.shields.io/badge/license-MIT-f4c430?style=flat-square)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/downloads/)
[![MCP](https://img.shields.io/badge/MCP-stdio-5c4ee5?style=flat-square)](#mcp)

**Speedup ceiling of one GPU step: time left if its hottest kernel cost nothing.**

Give stepceil one [Nsight Systems](https://developer.nvidia.com/nsight-systems) sqlite export of one NVTX range. It names the single place that blocked that step, and how much of the step the hottest kernel can still save. The command, the MCP tools, and `skill/SKILL.md` all read this same ledger. The skill does not recompute it.

![pressure is 80 ns of a 1000 ns step. The synchronize blocks 920 ns, so allow_rewrite is false.](docs/figures/flow.svg)

[Try it](#quick-start) · [The hot kernel](#the-hot-kernel-is-not-always-the-step) · [Overlap](#overlap-is-not-a-bigger-ceiling) · [Compare](#compare-two-captures) · [Agent](#agent) · [MCP](#mcp)

## Quick start

The repository includes four small exports, so this needs no GPU. Python 3.10 or newer. The program uses the standard library only.

```bash
git clone https://github.com/Tes-bo/stepceil.git
cd stepceil
python3 -m pip install .
stepceil examples/sync.sqlite --range step
```

From a checkout before install:

```bash
PYTHONPATH=src python3 -m stepceil examples/sync.sqlite --range step
```

On your own program, mark one iteration and export sqlite. `.nsys-rep` is not opened.

```bash
nsys profile --trace=cuda,nvtx -o report ./app
nsys export --type sqlite -o report.sqlite report.nsys-rep
stepceil report.sqlite --range step
```

```c
nvtxRangePushA("step");
/* one iteration */
nvtxRangePop();
```

`--json` prints the ledger an agent should read. The range name must occur once. Push/pop (NVTX type 59) and start/end (type 60) count. Marks do not. The match is exact.

## Which name marks the step

List the names before choosing one. A name can be accounted only when `count` is 1 and `closed` is 1. `closed` is how many of those events have an end timestamp after the start. `shortest_ns` and `longest_ns` are the durations of those closed occurrences. They do not pick a window. `inner` below occurs four times, from 4 ns to 16 ns, so it is not a step.

```bash
stepceil ranges examples/names.sqlite
```

```text
ranges: 2
name: inner
count: 4
closed: 4
shortest_ns: 4
longest_ns: 16
name: step
count: 1
closed: 1
shortest_ns: 16
longest_ns: 16
```

The MCP tool `step_ranges` takes `path` and returns that list as JSON.

## The hot kernel is not always the step

`pressure` runs for the first 80 ns of a 1000 ns step. `cudaDeviceSynchronize` covers the whole step, so the other 920 ns is that synchronize sitting on idle GPU time. Deleting `pressure` saves at most 80 ns. That is under 10% of the step, so `allow_rewrite` is false. The blocking piece of the synchronize starts at 80 ns, not at the start of the call.

![pressure is an 80 ns slice. The context synchronize blocks the other 920 ns, so the ceiling is under 10 percent.](docs/figures/sync-ceiling.svg)

`examples/sync.sqlite` is this step.

```text
range: step
step_start_ns: 0
step_end_ns: 1000
step_time_ns: 1000
gpu_busy_ns: 80
gpu_idle_ns: 920
assumption: single stream, no overlapping device work
longest_blocker.kind: sync
longest_blocker.name: context synchronize
longest_blocker.symbol: none
longest_blocker.start_ns: 80
longest_blocker.end_ns: 1000
longest_blocker.blocking_ns: 920
longest_blocker.source: none
longest_blocker.bytes: none
contained.kind: none
contained.name: none
contained.symbol: none
contained.blocking_ns: none
contained.contained.kind: none
contained.contained.name: none
contained.contained.symbol: none
contained.contained.blocking_ns: none
separate.kind: kernel
separate.name: pressure
separate.symbol: none
separate.start_ns: 0
separate.end_ns: 80
separate.blocking_ns: 80
separate.source: none
separate.bytes: none
separate.contained.kind: none
separate.contained.name: none
separate.contained.symbol: none
separate.contained.blocking_ns: none
separate.contained.contained.kind: none
separate.contained.contained.name: none
separate.contained.contained.symbol: none
separate.contained.contained.blocking_ns: none
further: 0
repeated: 0
short_idle_ns: 0
short_idle_longest_ns: 0
hottest_kernel.name: pressure
hottest_kernel.symbol: none
hottest_kernel.ceiling_ns: 80
hottest_kernel.instances: 1
hottest_kernel.source: none
ceiling_ns: 80
allow_rewrite: false
ceiling_refused: none
overlap.overlap_ns: none
overlap.first.kind: none
overlap.first.name: none
overlap.first.start_ns: none
overlap.first.end_ns: none
overlap.first.stream_id: none
overlap.first.context_id: none
overlap.first.bytes: none
overlap.second.kind: none
overlap.second.name: none
overlap.second.start_ns: none
overlap.second.end_ns: none
overlap.second.stream_id: none
overlap.second.context_id: none
overlap.second.bytes: none
rewrite_threshold_numerator: 1
rewrite_threshold_denominator: 10
decision: change context synchronize first
```

```json
{
  "range": "step",
  "step_start_ns": 0,
  "step_end_ns": 1000,
  "step_time_ns": 1000,
  "gpu_busy_ns": 80,
  "gpu_idle_ns": 920,
  "assumption": "single stream, no overlapping device work",
  "longest_blocker": {
    "kind": "sync",
    "name": "context synchronize",
    "symbol": null,
    "start_ns": 80,
    "end_ns": 1000,
    "blocking_ns": 920,
    "source": null,
    "bytes": null
  },
  "contained": null,
  "separate": {
    "kind": "kernel",
    "name": "pressure",
    "symbol": null,
    "start_ns": 0,
    "end_ns": 80,
    "blocking_ns": 80,
    "source": null,
    "bytes": null,
    "contained": null
  },
  "further": [],
  "repeated": [],
  "short_idle_ns": 0,
  "short_idle_longest_ns": 0,
  "hottest_kernel": {
    "name": "pressure",
    "symbol": null,
    "ceiling_ns": 80,
    "instances": 1,
    "source": null
  },
  "ceiling_ns": 80,
  "allow_rewrite": false,
  "ceiling_refused": null,
  "overlap": null,
  "rewrite_threshold_numerator": 1,
  "rewrite_threshold_denominator": 10,
  "decision": "change context synchronize first"
}
```

## Overlap is not a bigger ceiling

`alpha` and `beta` each run 60 ns, on two streams, and overlap. Adding them produces 120 ns, which is longer than the 100 ns step. The GPU was busy for the union, 100 ns. This version withholds the ceiling instead of publishing the inflated sum. `ceiling_ns` is null and `allow_rewrite` is false. Busy and idle are still reported. `overlap` names the two clipped launches: `alpha` on stream 1 and `beta` on stream 2. `overlap.overlap_ns` is their intersection, 20 ns. Do not add the two durations. `bytes` on each side is that row's size when the side is a copy or memset that lies entirely inside the step. A kernel, a missing size, or a transfer that crosses the boundary is null. Do not add the two sizes.

![Two 60 ns kernels overlap, so their sum is 120 ns and the union is 100 ns. The ceiling is withheld.](docs/figures/overlap-withheld.svg)

`examples/overlap.sqlite` is this step.

```text
range: step
step_start_ns: 0
step_end_ns: 100
step_time_ns: 100
gpu_busy_ns: 100
gpu_idle_ns: 0
assumption: overlapping device work on more than one stream
longest_blocker.kind: none
longest_blocker.name: none
longest_blocker.symbol: none
longest_blocker.start_ns: none
longest_blocker.end_ns: none
longest_blocker.blocking_ns: none
longest_blocker.source: none
longest_blocker.bytes: none
contained.kind: none
contained.name: none
contained.symbol: none
contained.blocking_ns: none
contained.contained.kind: none
contained.contained.name: none
contained.contained.symbol: none
contained.contained.blocking_ns: none
separate.kind: none
separate.name: none
separate.symbol: none
separate.start_ns: none
separate.end_ns: none
separate.blocking_ns: none
separate.source: none
separate.bytes: none
separate.contained.kind: none
separate.contained.name: none
separate.contained.symbol: none
separate.contained.blocking_ns: none
separate.contained.contained.kind: none
separate.contained.contained.name: none
separate.contained.contained.symbol: none
separate.contained.contained.blocking_ns: none
further: none
repeated: none
short_idle_ns: none
short_idle_longest_ns: none
hottest_kernel.name: none
hottest_kernel.symbol: none
hottest_kernel.ceiling_ns: none
hottest_kernel.instances: none
hottest_kernel.source: none
ceiling_ns: none
allow_rewrite: false
ceiling_refused: overlapping device work on more than one stream
overlap.overlap_ns: 20
overlap.first.kind: kernel
overlap.first.name: alpha
overlap.first.start_ns: 0
overlap.first.end_ns: 60
overlap.first.stream_id: 1
overlap.first.context_id: 1
overlap.first.bytes: none
overlap.second.kind: kernel
overlap.second.name: beta
overlap.second.start_ns: 40
overlap.second.end_ns: 100
overlap.second.stream_id: 2
overlap.second.context_id: 1
overlap.second.bytes: none
rewrite_threshold_numerator: 1
rewrite_threshold_denominator: 10
decision: do not rewrite a kernel from this report
```

## Compare two captures

After a change, export the same NVTX range again. `faster` is true only when the later step is strictly shorter. A withheld ceiling still has a step time, so the comparison stands.

`examples/after.sqlite` is the same kernel with the synchronize cut from 1000 ns to 200 ns. The step is 800 ns shorter. `pressure` is still 80 ns, which is now at least 10% of the shorter step, so `allow_rewrite` becomes true. The synchronize still blocks the other 120 ns. Both ledgers say `change context synchronize first`. The flag flipping does not change the action.

![The step shrinks from 1000 ns to 200 ns. pressure stays 80 ns, and the synchronize is still the longest blocker.](docs/figures/diff.svg)

```bash
stepceil diff examples/sync.sqlite examples/after.sqlite --range step
```

```text
range: step
before_step_time_ns: 1000
after_step_time_ns: 200
step_time_ns_delta: -800
faster: true
before_longest_blocker.kind: sync
before_longest_blocker.name: context synchronize
before_longest_blocker.symbol: none
before_longest_blocker.start_ns: 80
before_longest_blocker.end_ns: 1000
before_longest_blocker.blocking_ns: 920
before_longest_blocker.source: none
before_longest_blocker.bytes: none
after_longest_blocker.kind: sync
after_longest_blocker.name: context synchronize
after_longest_blocker.symbol: none
after_longest_blocker.start_ns: 80
after_longest_blocker.end_ns: 200
after_longest_blocker.blocking_ns: 120
after_longest_blocker.source: none
after_longest_blocker.bytes: none
before_ceiling_ns: 80
after_ceiling_ns: 80
before_allow_rewrite: false
after_allow_rewrite: true
before_ceiling_refused: none
after_ceiling_refused: none
before_decision: change context synchronize first
after_decision: change context synchronize first
```

The MCP tool `step_diff` takes `before`, `after`, and `range` and returns that same JSON.

## What the numbers mean

Device intervals are clipped to the NVTX range first. `gpu_busy_ns` is the length of the union of clipped kernel, memcpy, and memset intervals. `gpu_idle_ns` is `step_time_ns - gpu_busy_ns`.

When nothing on the device overlaps:

- Each clipped launch blocks the step for its own duration.
- The ceiling of a kernel name is the sum of its clipped launches, not the longest one.
- A synchronization call blocks the step only where it covers idle time. Time already covered by a kernel, copy, or memset stays on that operation. Another synchronization that overlaps the same idle does not shorten that piece. The longest blocker is the longest piece, not the sum of the pieces. When several calls cover that same piece, the narrowest call is the blocker. A wider range that only wraps it is not a longer blocker. `contained` is the longest operation strictly inside that piece. It is not a second decision. An equal span is the same piece, not something inside it. `contained.contained` opens that result once more, then stops. `separate` is the longest piece that does not meet the blocker. Touching endpoints do not meet. It is not a second decision. `further` lists each later piece that does not meet those already listed and that lasts at least a tenth of the step, the same bar as `allow_rewrite`. A shorter piece is left off. `further` is not a decision either. `repeated` lists a host API, and an NVTX range whose every idle piece is shorter than a tenth of the step, when at least two calls together last that long. A single call is not repeated, even when a kernel splits it. A range that already has one idle piece that long is not listed again. A range whose idle sits inside another listed range is left off. When two ranges cover the same idle, the narrower source span remains. Do not add `idle_ns` to a blocker, do not add two `repeated` entries together, and do not read `idle_ns` as the blocking time of one call. `short_idle_ns` is idle that the blocker, `separate`, and `further` do not cover. `short_idle_longest_ns` is the longest contiguous stretch of that idle. Several calls can tile one stretch, so the stretch is not a call and not a blocker. Do not add `short_idle_ns` to a blocker or to a `repeated` entry.
- Idle time with no synchronization call, no host API call, and no other NVTX range is a gap named `gpu idle`. A `CUPTI_ACTIVITY_KIND_RUNTIME` call and any NVTX range other than the step itself block only the idle they cover, under their own names. Neither is device work, so neither changes `gpu_busy_ns` or overlaps a kernel. The step range is not a blocker: it covers the whole window. A call or range from another process is not this step's blocker.

`allow_rewrite` is true only when `ceiling_ns * 10 >= step_time_ns`. The threshold is fixed at 1/10 of the step and is repeated in the ledger. `decision` is the action. Rewrite a kernel only when it says `rewrite` and that kernel's name. `allow_rewrite` can be true while `decision` still says to change a synchronize or a gap.

Overlap on different streams, or different CUDA contexts, is reported as more than one stream. Overlap on one stream is reported separately, because a single stream should not do that. `overlap` names the clipped pair of that class with the longest overlap. `overlap.overlap_ns` is that intersection. Ties keep the earlier one. The pair does not follow the order of rows in the export. A cross-stream pair is kept when both kinds of overlap are present, even when a same-stream overlap is longer. Adding the two durations is not a ceiling. Intervals are half-open: a launch that ends at the timestamp another starts does not overlap it.

Times are nanoseconds. `source` is null. These sqlite tables do not carry a file and line. `longest_blocker.bytes`, `overlap.first.bytes`, and `overlap.second.bytes` are the size of a copy or memset when that row has a `bytes` column and the transfer lies entirely inside the step. A kernel, synchronize, gap, a missing size, or a transfer that crosses the NVTX boundary is null: the column counts the whole transfer, not the clipped piece. Do not invent one, and do not add the two overlap sizes.

## Input

| Table | Required | Role |
| --- | --- | --- |
| `NVTX_EVENTS` | yes | the step window |
| `StringIds` | yes | range text and kernel names |
| `CUPTI_ACTIVITY_KIND_KERNEL` | no | kernel launches; a missing table means none |
| `CUPTI_ACTIVITY_KIND_MEMCPY` | no | copies |
| `CUPTI_ACTIVITY_KIND_MEMSET` | no | memsets |
| `CUPTI_ACTIVITY_KIND_SYNCHRONIZATION` | no | host synchronization calls |
| `CUPTI_ACTIVITY_KIND_RUNTIME` | no | host API calls on idle; a missing table means none |

Kernel names come from `shortName`, then `demangledName`, then `mangledName`. `symbol` is `mangledName`, then `demangledName`, or null when the export has neither. A short name drops template arguments, so two kernels can share it. The ceiling sums launches with the same name and the same symbol, and `rewrite` needs the longest blocker to be that same symbol. NVTX text is the `text` column when it is non-empty, otherwise `StringIds` at `textId`.

Memcpy and synchronization names come from `ENUM_CUDA_MEMCPY_OPER` and `ENUM_CUPTI_SYNC_TYPE` when those tables are in the export. Nsight Systems 2022.4 numbers memcpy kinds differently from older CUPTI headers, so the file's label wins. Exports without those tables keep stepceil's short names, such as `memcpy HtoD` and `context synchronize`.

That label stays, unless one `CUPTI_ACTIVITY_KIND_RUNTIME` call with the same `correlationId` covers the record and its name contains `Sync` or `Wait`. Covering means the call starts at or before the record and ends at or after it. A launch that only overlaps the record does not rename it. Nsight Systems 2022.4 labels `cudaDeviceSynchronize` as `Unknown` and puts the API name on the covering runtime call.

More than one GPU inside the step exits 2 and prints no ledger. A peer copy counts `srcDeviceId` and `dstDeviceId` as well as `deviceId`. A null peer column is not a GPU. Work that does not meet the NVTX range is ignored, including a peer copy that lies entirely outside it.

More than one process inside the step exits 2 and prints no ledger. Nsight stores that identity in `globalPid` on device rows and on synchronization records. The step range counts as well: its `globalTid` with the low 24 bits cleared is the step's process, so a step whose device work came only from another process is refused instead of accounted. A null value is not a second process. Stream ids are reused across processes, so another process's kernels are not this step's stream and not its ceiling. A process whose records lie entirely outside the range is ignored. `correlationId` is also reused across processes. A runtime call renames a synchronization only when `globalTid` with its low 24 bits cleared equals that record's `globalPid`.

## Agent

`skill/SKILL.md` is the rule sheet:

- Obey `decision`. Rewrite a kernel only when it is `rewrite` followed by that kernel's name.
- Any other `decision` means do not rewrite a kernel from this report. When `overlap` is set, those two operations collided. Do not add their durations.
- After a change, run `stepceil diff` on the two exports. Do not claim the step got faster unless `faster` is true.

## MCP

`stepceil mcp` speaks [MCP](https://modelcontextprotocol.io) on stdin and stdout. `step_ceiling` takes `path` and `range` and returns the `--json` ledger. `step_diff` compares two exports. `step_ranges` lists range names. A refused report is a tool error.

```json
{
  "mcpServers": {
    "stepceil": {
      "command": "stepceil",
      "args": ["mcp"]
    }
  }
}
```

Cursor, and any other client that launches a stdio MCP server, uses that block.

## What this version does not do

- It does not profile or rewrite kernels.
- It does not report occupancy or memory throughput.
- It does not account more than one GPU.
- It does not account more than one process.
- It does not invent a ceiling for overlapping device work.

## Exit status

| Code | Meaning |
| --- | --- |
| 0 | A ledger was printed. A withheld ceiling is still a ledger. |
| 2 | The file is not one accountable step, or `stepceil mcp` was given arguments. |

The refusal line goes to stderr and starts with `stepceil:`.

## Develop

```bash
python3 examples/build_examples.py
python3 -m unittest discover -s tests -v
```

## License

[MIT](LICENSE).

# stepceil

Speedup ceiling of one GPU step: time left if its hottest kernel cost nothing.

stepceil reads one [Nsight Systems](https://developer.nvidia.com/nsight-systems) sqlite export and accounts a single NVTX range. It reports the step time, how long the GPU was busy, the one place that blocked the step the longest, and the ceiling of the hottest kernel. The ceiling is the time that kernel still occupies after its launches are clipped to the step. Rewriting that kernel cannot make the step shorter than this ceiling, and it saves nothing like that much when the kernel is not what blocked the step.

The command, the `step_ceiling` MCP tool, and `skill/SKILL.md` all use this same ledger. The skill tells an agent when it may rewrite the kernel. It does not recompute the numbers.

## What this version does not do

- It does not open a `.nsys-rep`. Export sqlite first.
- It does not profile, rewrite kernels, or compare two reports. Run it again on a new capture.
- It does not report occupancy, memory throughput, source file, or source line. `source` is null.
- It does not account more than one GPU.
- It does not give a ceiling when device work inside the step overlaps. Summing those durations would overstate the time the kernel can save.

## Install

```bash
pip install .
stepceil --help
```

From a checkout, without installing:

```bash
PYTHONPATH=src python3 -m stepceil --help
```

Python 3.10 or newer. The program uses the standard library only.

## Prepare a report

Mark one step. The range name must occur once in the export. Push/pop ranges (NVTX event type 59) and start/end ranges (event type 60) count. Marks do not. The match is exact, including case.

```c
nvtxRangePushA("step");
/* one iteration */
nvtxRangePop();
```

```bash
nsys profile --trace=cuda,nvtx -o report ./app
nsys export --type sqlite -o report.sqlite report.nsys-rep
stepceil report.sqlite --range step
```

`--json` prints the same ledger as JSON. That is the form an agent should read.

Tables this version reads:

| Table | Required | Role |
| --- | --- | --- |
| `NVTX_EVENTS` | yes | the step window |
| `StringIds` | yes | range text and kernel names |
| `CUPTI_ACTIVITY_KIND_KERNEL` | no | kernel launches; a missing table means none |
| `CUPTI_ACTIVITY_KIND_MEMCPY` | no | copies |
| `CUPTI_ACTIVITY_KIND_MEMSET` | no | memsets |
| `CUPTI_ACTIVITY_KIND_SYNCHRONIZATION` | no | host synchronization calls |

A missing optional table means there were no events of that kind. Kernel names come from `shortName`, then `demangledName`, then `mangledName`. NVTX text is the `text` column when it is non-empty, otherwise `StringIds` at `textId`.

Times in the ledger are nanoseconds. Intervals are half-open: a launch that ends at the same timestamp another starts does not overlap it.

## How the numbers are computed

Device intervals are clipped to the NVTX range first. `gpu_busy_ns` is the length of the union of clipped kernel, memcpy, and memset intervals. `gpu_idle_ns` is `step_time_ns - gpu_busy_ns`. Those two add up to the step.

On a single stream with no overlapping device work:

- Each clipped launch blocks the step for its own duration.
- The ceiling of a kernel name is the sum of its clipped launches, not the longest one.
- A synchronization call blocks the step only where it covers idle time. Time already covered by a kernel, copy, or memset stays on that operation.
- Idle time that no synchronization call covers is a gap named `gpu idle`.

`allow_rewrite` is true only when `ceiling_ns * 10 >= step_time_ns`. The threshold is fixed in this version at 1/10 of the step. The ledger repeats it as `rewrite_threshold_numerator` and `rewrite_threshold_denominator`.

When two clipped device intervals overlap, the ceiling is withheld. `ceiling_ns` is null, `allow_rewrite` is false, and `ceiling_refused` says why. Busy and idle are still reported, using the union, so overlapped launches are not added twice. Overlap on different streams, or different CUDA contexts, is "more than one stream". Overlap on one stream is reported separately because a single stream should not do that.

More than one `deviceId` inside the step is refused before any of this. The process exits 2 and prints no ledger.

## A step whose hottest kernel is the wrong target

The kernel `pressure` runs for 80 ns at the start of a 1000 ns step. `cudaDeviceSynchronize` covers the whole step, so the other 920 ns of the step is the synchronize sitting on idle time. Perfecting `pressure` saves at most 80 ns, which is under 10%, so `allow_rewrite` is false. The longest blocker is the synchronize, and its blocking interval starts at 80 ns, not at the start of the call.

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
longest_blocker.start_ns: 80
longest_blocker.end_ns: 1000
longest_blocker.blocking_ns: 920
longest_blocker.source: none
hottest_kernel.name: pressure
hottest_kernel.ceiling_ns: 80
hottest_kernel.instances: 1
hottest_kernel.source: none
ceiling_ns: 80
allow_rewrite: false
ceiling_refused: none
rewrite_threshold_numerator: 1
rewrite_threshold_denominator: 10
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
    "start_ns": 80,
    "end_ns": 1000,
    "blocking_ns": 920,
    "source": null
  },
  "hottest_kernel": {
    "name": "pressure",
    "ceiling_ns": 80,
    "instances": 1,
    "source": null
  },
  "ceiling_ns": 80,
  "allow_rewrite": false,
  "ceiling_refused": null,
  "rewrite_threshold_numerator": 1,
  "rewrite_threshold_denominator": 10
}
```

## Overlapping streams

Two kernels overlap on stream 1 and stream 2. Their durations would add to 120 ns, which is longer than the step. The union is 100 ns, and the ceiling is withheld.

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
longest_blocker.start_ns: none
longest_blocker.end_ns: none
longest_blocker.blocking_ns: none
longest_blocker.source: none
hottest_kernel.name: none
hottest_kernel.ceiling_ns: none
hottest_kernel.instances: none
hottest_kernel.source: none
ceiling_ns: none
allow_rewrite: false
ceiling_refused: overlapping device work on more than one stream
rewrite_threshold_numerator: 1
rewrite_threshold_denominator: 10
```

## MCP

`stepceil mcp` speaks [MCP](https://modelcontextprotocol.io) on stdin and stdout. It has one tool, `step_ceiling`, with string arguments `path` and `range`. The tool returns the `--json` ledger. A refused report is a tool error, not a second kind of arithmetic.

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

## Skill

`skill/SKILL.md` is the rule sheet for a coding agent:

- Change `longest_blocker` first.
- Rewrite the hottest kernel only when `allow_rewrite` is true and that kernel is the longest blocker.
- When `ceiling_refused` is not null, do not rewrite kernels from that report.
- After a change, capture the same range again and show both ledgers. Do not claim the step got faster without the second report.

## Exit status

| Code | Meaning |
| --- | --- |
| 0 | A ledger was printed. A withheld ceiling is still a ledger. |
| 2 | The file is not one accountable step, or `stepceil mcp` was given arguments. |

The refusal line is written to stderr and starts with `stepceil:`.

## Develop

```bash
python3 -m unittest discover -s tests -v
```

## License

[MIT](LICENSE).

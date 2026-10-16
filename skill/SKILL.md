---
name: stepceil
description: >
  Account one GPU step from an Nsight Systems sqlite export before changing
  a CUDA kernel. Use when a profile should say how much of the step the
  hottest kernel can still save, when the NVTX range name in an export is
  not known, or when an agent is about to rewrite a kernel to make a step
  faster.
---

# stepceil

Run stepceil on one Nsight Systems sqlite export. Do not estimate the ceiling yourself.

If the range name is not already known, list the names in the export first. Do not guess one, and do not replace a missing range with a start and end time.

```
stepceil ranges REPORT.sqlite --json
```

`step_ranges` is that list when MCP is connected. Pass `path`. Account a name only when `count` is 1 and `closed` is 1. `shortest_ns` and `longest_ns` are how long the closed occurrences were. They do not choose a window. A name that occurs more than once is not a step, even when those two numbers match.

```
stepceil REPORT.sqlite --range STEP --json
```

`STEP` is that NVTX range name. `REPORT.sqlite` is the file from `nsys export --type sqlite`, not a `.nsys-rep`.

If the `step_ceiling` MCP tool is connected, call it with `path` and `range`. It returns this same JSON.

Obey `decision`. Do not derive a different action from `allow_rewrite`.

- Rewrite a kernel only when `decision` is `rewrite` followed by that kernel's name.
- Any other `decision` means do not rewrite a kernel from this report. `change ... first` names the blocker to change, including when `allow_rewrite` is true.
- When `decision` is `do not rewrite a kernel from this report`, profile a step whose device work does not overlap, then run the program again. `overlap` names the two clipped operations with the longest overlap of that class. `overlap.overlap_ns` is that intersection. Do not add their durations. A shorter collision is not the pair. Reading the export again does not change the pair.
- If the command reports more than one process, capture one process. Stream ids from separate processes are not one stream. Do not look for a same-stream overlap. A runtime call from another process does not name a synchronization.
- `source` is null in this version. Do not invent a file and line.
- When the blocker is a copy or memset, `longest_blocker.bytes` is its size from the export only if that transfer lies entirely inside the step. Null means the export did not say, or the transfer crosses the boundary. Do not invent a size.
- A synchronization's blocking time is the idle that call covers. Another synchronization on the same idle does not shorten it. A kernel, copy, or memset still splits it. Do not add the pieces.
- After a change, capture the same NVTX range again and compare the two exports. Do not claim the step got faster unless `faster` is true.

```
stepceil diff BEFORE.sqlite AFTER.sqlite --range STEP --json
```

`step_diff` is the same comparison when MCP is connected. Pass `before`, `after`, and `range`.

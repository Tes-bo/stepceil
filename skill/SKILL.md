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

`step_ranges` is that list when MCP is connected. Pass `path`. Account a name only when `count` is 1 and `closed` is 1.

```
stepceil REPORT.sqlite --range STEP --json
```

`STEP` is that NVTX range name. `REPORT.sqlite` is the file from `nsys export --type sqlite`, not a `.nsys-rep`.

If the `step_ceiling` MCP tool is connected, call it with `path` and `range`. It returns this same JSON.

Obey the JSON:

- Change `longest_blocker` first.
- Rewrite the hottest kernel only when `allow_rewrite` is true and that kernel is the longest blocker.
- When `allow_rewrite` is false, do not rewrite the hottest kernel. The program sets that flag when the ceiling is under 10% of the step (`ceiling_ns * 10 >= step_time_ns`).
- When `ceiling_refused` is not null, do not rewrite any kernel from this report. Profile a step whose device work does not overlap, then run the program again.
- `source` is null in this version. Do not invent a file and line.
- After a change, capture the same NVTX range again and compare the two exports. Do not claim the step got faster unless `faster` is true.

```
stepceil diff BEFORE.sqlite AFTER.sqlite --range STEP --json
```

`step_diff` is the same comparison when MCP is connected. Pass `before`, `after`, and `range`.

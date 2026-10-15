---
name: stepceil
description: >
  Account one GPU step from an Nsight Systems sqlite export before changing
  a CUDA kernel. Use when a profile should say how much of the step the
  hottest kernel can still save, or when an agent is about to rewrite a
  kernel to make a step faster.
---

# stepceil

Run stepceil on one Nsight Systems sqlite export. Do not estimate the ceiling yourself.

```
stepceil REPORT.sqlite --range STEP --json
```

`STEP` is the NVTX range name, and it must occur once. `REPORT.sqlite` is the file from `nsys export --type sqlite`, not a `.nsys-rep`.

If the `step_ceiling` MCP tool is connected, call it with `path` and `range`. It returns this same JSON.

Obey the JSON:

- Change `longest_blocker` first.
- Rewrite the hottest kernel only when `allow_rewrite` is true and that kernel is the longest blocker.
- When `allow_rewrite` is false, do not rewrite the hottest kernel. The program sets that flag when the ceiling is under 10% of the step (`ceiling_ns * 10 >= step_time_ns`).
- When `ceiling_refused` is not null, do not rewrite any kernel from this report. Profile a step whose device work does not overlap, then run the program again.
- `source` is null in this version. Do not invent a file and line.
- After a change, capture the same NVTX range again and run the program again. Show both `step_time_ns` values and both longest blockers. Do not claim the step got faster without the second report.

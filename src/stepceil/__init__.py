"""Speedup ceiling of one GPU step."""

VERSION = "0.1.0"

# allow_rewrite is true only when ceiling_ns * REWRITE_DENOMINATOR >= step_time_ns.
REWRITE_NUMERATOR = 1
REWRITE_DENOMINATOR = 10

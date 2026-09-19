# Milestone 3: Bounded Single-Row Linear Parallelism

## Baseline and measurement

The parallel executor extends the cached session's single-row linear path.
The full-prefix path remains the independent correctness oracle; the scalar
cached path remains callable and provides the performance baseline.

The recorded `perf` samples put 95.97–99.21% of sampled user-mode
CPU time in `linear`, depending on workload. These are function self-sample
percentages, not fractions of wall time or a proof of memory-bandwidth limits.
The toolchain already emits some SIMD instructions. Hardware counter events
were unavailable in WSL.

A Release build measured the five real GPT-2 124M single-row shapes
with GCC 15.2, CMake Release, one warmup, five samples, and eight calls per
sample. The medians below are per call, using the first layer's actual weights:

| Shape | Input × output | Weight floats | Scalar median (ms) |
|---|---:|---:|---:|
| QKV | 768 × 2304 | 1,769,472 | 0.750 |
| attention projection | 768 × 768 | 589,824 | 0.245 |
| FC | 768 × 3072 | 2,359,296 | 0.988 |
| FC projection | 3072 × 768 | 2,359,296 | 1.105 |
| tied-vocabulary projection | 768 × 50257 | 38,597,376 | 17.992 |

Twelve copies of the four layer shapes plus the vocabulary shape sum to about
55.0 ms, near the independently measured 57.5 ms one-token cached decode. The
sum is an estimate across different input vectors, not an exact end-to-end
breakdown. This confirms a narrow linear-focused experiment. Because current
prefill invokes the same one-row core for every token, no multi-row prefill
optimization is assumed.

The pre-change scalar benchmark at P=3, G=8, capacity=1024 gave
cached generation samples (ms) `576.833, 572.249, 576.075, 574.306, 569.844`
and median 574.306. Prefill median was 168.680 ms; one decode median was
57.474 ms. These are scalar baseline measurements, not parallel gains.

## Executor design

Add one small, synchronous persistent executor specialized for one-row FP32
linear. `total_threads` includes the submitting thread. Each job partitions
whole output channels into disjoint contiguous ranges; each channel's dot
product uses the original scalar `kernels::linear` accumulation order.
No input-channel splitting or floating reduction tree is introduced, so
bit-level parity remains a testable goal.

The executor owns its worker threads and serializes simultaneous submissions.
Calls are synchronous: spans borrowed for a job need only remain valid until
the call returns. A session does not retain an executor reference; a caller
passes one to `prefill`/`decode`. M1 never calls the executor. One-thread and
small jobs use the scalar kernel directly. The small-job threshold is 65,536
weight floats per requested thread. It is a simple overhead guard, not an
adaptive scheduler; measurements support leaving the feature opt-in.

Job publication and completion use a mutex, condition variables, and a job
generation number. Workers write disjoint output slices. Before returning, the
submitter waits for every worker, then propagates any captured exception.
Destruction stops, wakes, and joins workers; a partially failed constructor
must do the same. Destruction is not concurrent with a submission.

This is not a generic task scheduler: no arbitrary jobs, futures, async queue,
work stealing, nested submission, or parallel attention. There are no SIMD
intrinsics, BLAS calls, multi-request batching, or block-prefill changes.

## Thread-count selection

The tested counts were 1, 2, 4, 8, and 16 total threads. Eight was best for
the primary workload; 16 regressed, and per-shape gains are uneven. The API
therefore remains explicit/opt-in with one thread as its default. The complete
commands, raw samples, numeric checks, and sanitizer outcomes are in
[`MILESTONE3_RESULTS.md`](MILESTONE3_RESULTS.md). This result is local to the
measured CPU and build and does not justify a portable default thread count.

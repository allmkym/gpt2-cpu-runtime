# Milestone 3 Measurement and Validation Record

Measured on 2026-09-17. The scalar baseline was recorded before the parallel
changes in the same checkout. The thread-count sweep used one Release build;
a final 1/8-thread recheck after serializing scalar fallback is recorded
separately below. Results from different runs are labeled rather than merged.

## Environment and commands

- CPU: Intel Core i7-14700HX; WSL2 kernel
  `6.6.114.1-microsoft-standard-WSL2`; 28 logical CPUs visible.
- Compiler: GCC 15.2.0; CMake Release, C++20, `-O3 -DNDEBUG`, project warnings
  and runtime `-ffp-contract=off`. No sanitizer build was timed.
- Checkpoint: GPT-2 124M, llm.c-format version 1. The paths in the commands
  below are supplied through `CHECKPOINT` and `LLMC_DIR` environment variables.
- `--threads` counts the caller as one thread. The pool is constructed before
  the timed runs. Model load, session creation, logging, and dumps are excluded
  from generation timings. One warmup precedes each measured metric. Samples
  are wall-clock milliseconds in execution order, not confidence intervals.
- Runs were sequential, not CPU-pinned or interleaved. Machine load and cache
  state can change across configurations; this is a bounded local result.

Build and validation commands:

```bash
export CHECKPOINT=/path/to/gpt2_124M.bin
cmake -S . -B build-m3-release -DCMAKE_BUILD_TYPE=Release \
  -DGPT2_TEST_CHECKPOINT="$CHECKPOINT"
cmake --build build-m3-release --parallel
ctest --test-dir build-m3-release -N
ctest --test-dir build-m3-release --output-on-failure

cmake -S . -B build-m3-asan -DCMAKE_BUILD_TYPE=Debug \
  -DGPT2_ENABLE_SANITIZERS=ON \
  -DCMAKE_CXX_FLAGS_DEBUG=-D_GLIBCXX_ASSERTIONS \
  -DGPT2_TEST_CHECKPOINT="$CHECKPOINT"
cmake --build build-m3-asan --parallel
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 \
UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
  ctest --test-dir build-m3-asan --output-on-failure

cmake -S . -B build-m3-tsan -DCMAKE_BUILD_TYPE=Debug \
  -DGPT2_ENABLE_TSAN=ON \
  -DGPT2_TEST_CHECKPOINT="$CHECKPOINT"
cmake --build build-m3-tsan --parallel
ctest --test-dir build-m3-tsan --output-on-failure
```

Release CTest registered six tests including `gpt2_real_checkpoint`, and 6/6
passed in 4.31 s on the final test version. ASan/LSan/UBSan with
`_GLIBCXX_ASSERTIONS` passed 6/6 in 54.30 s, with no finding. GCC TSan was run
separately: the final suite, including the concurrent scalar-fallback test,
passed 6/6 in 165.45 s without a race report; its real-checkpoint test took
163.06 s. The preceding full TSan run had also passed 6/6 in 160.98 s. These
are observed runs, not a claim that sanitizers prove absence of all defects.

## Single-row kernel measurements

Command for each total thread count `N` in `1 2 4 8 16`:

```bash
for N in 1 2 4 8 16; do
  ./build-m3-release/gpt2_linear_benchmark \
    "$CHECKPOINT" 1 5 8 "$N"
done
```

Each sample is eight calls to the same shape divided by eight; the input is
fixed and the weights come from the checkpoint's first layer (or tied token
embedding for vocabulary projection). The sum estimate is 12 times the four
layer-shape medians plus the vocabulary median; it is not a measured decode.
The first pre-change scalar shape study gave medians 0.750, 0.245, 0.988,
1.105, and 17.992 ms respectively. The following complete sweep was a later,
separate run of the M3 build.

| Total threads | Shape (input × output) | Five samples, ms | Median, ms |
|---:|---|---|---:|
| 1 | QKV (768 × 2304) | .730, .753, .769, .742, .799 | .753 |
| 1 | attention projection (768 × 768) | .242, .243, .261, .256, .244 | .244 |
| 1 | FC (768 × 3072) | .986, 1.114, 1.009, 1.014, 1.056 | 1.014 |
| 1 | FC projection (3072 × 768) | 1.109, 1.056, 1.151, 1.056, 1.134 | 1.109 |
| 1 | vocabulary (768 × 50257) | 17.846, 17.638, 18.123, 17.852, 17.960 | 17.852 |
| 2 | QKV | .417, .442, .420, .399, .415 | .417 |
| 2 | attention projection | .151, .149, .149, .169, .148 | .149 |
| 2 | FC | .575, .529, .538, .559, .580 | .559 |
| 2 | FC projection | .567, .585, .572, .603, .589 | .585 |
| 2 | vocabulary | 9.354, 9.324, 9.359, 9.284, 9.099 | 9.324 |
| 4 | QKV | .264, .266, .250, .288, .232 | .264 |
| 4 | attention projection | .127, .108, .142, .129, .120 | .127 |
| 4 | FC | .343, .328, .336, .323, .360 | .336 |
| 4 | FC projection | .382, .372, .381, .348, .344 | .372 |
| 4 | vocabulary | 5.274, 5.243, 5.029, 5.242, 5.120 | 5.242 |
| 8 | QKV | .228, .253, .226, .256, .228 | .228 |
| 8 | attention projection | .118, .120, .162, .120, .119 | .120 |
| 8 | FC | .314, .320, .309, .320, .312 | .314 |
| 8 | FC projection | .297, .311, .344, .353, .311 | .311 |
| 8 | vocabulary | 3.429, 3.457, 3.191, 3.183, 3.012 | 3.191 |
| 16 | QKV | .239, .268, .223, .263, .265 | .263 |
| 16 | attention projection | .248, .241, .245, .252, .258 | .248 |
| 16 | FC | .292, .283, .283, .256, .268 | .283 |
| 16 | FC projection | .319, .288, .263, .284, .282 | .284 |
| 16 | vocabulary | 3.076, 2.899, 2.725, 2.694, 2.700 | 2.725 |

Estimated linear time per token by thread count: 1 = 55.301 ms, 2 = 29.844,
4 = 18.418, 8 = 14.872, 16 = 15.656. At 16 threads, the attention projection
falls under the small-job threshold and uses the scalar kernel. Increasing
thread count therefore does not monotonically improve the whole decode.

## Primary end-to-end workload: P=3, G=8

For each `N` in `1 2 4 8 16`:

```bash
for N in 1 2 4 8 16; do
  ./build-m3-release/gpt2_benchmark \
    --checkpoint "$CHECKPOINT" \
    --tokens 15496,11,616 --generation-tokens 8 --capacity 1024 \
    --warmup 1 --repetitions 5 --threads "$N"
done
```

The scalar M2 run before code changes measured cached-generation samples
`576.833, 572.249, 576.075, 574.306, 569.844` (median 574.306 ms),
prefill median 168.680 ms, decode median 57.474 ms, and full-prefix median
2913.146 ms. This pre-change run is separate from the M3-build sweep and the
earlier M2 full/cached measurement reported in [implementation notes](../IMPLEMENTATION_NOTES.md).

All M3-build raw samples below are milliseconds; each row is one independently
timed metric with one warmup and five repetitions. `projection_all` and
`projection_last` are scalar attribution probes, not part of the parallel
runtime optimization.

| Threads | Metric | Five samples | Median |
|---:|---|---|---:|
| 1 | session creation | 22.304, 22.392, 22.720, 22.635, 22.159 | 22.392 |
| 1 | prefill | 171.372, 170.172, 170.851, 169.983, 169.260 | 170.172 |
| 1 | single decode | 57.144, 57.150, 57.011, 57.323, 59.267 | 57.150 |
| 1 | full-prefix generation | 2962.722, 2988.028, 2987.197, 3016.010, 3003.745 | 2988.028 |
| 1 | cached generation | 589.569, 578.609, 603.485, 599.313, 604.665 | 599.313 |
| 1 | projection all | 54.660, 56.321, 54.752, 56.436, 55.336 | 55.336 |
| 1 | projection last | 17.953, 18.155, 18.982, 19.364, 19.624 | 18.982 |
| 2 | session creation | 22.946, 22.565, 23.418, 23.330, 23.656 | 23.330 |
| 2 | prefill | 97.646, 97.606, 96.337, 97.378, 93.672 | 97.378 |
| 2 | single decode | 31.896, 32.766, 31.870, 32.323, 31.813 | 31.896 |
| 2 | full-prefix generation | 2978.028, 2941.345, 2951.783, 2955.647, 2959.070 | 2955.647 |
| 2 | cached generation | 320.212, 323.942, 320.296, 313.163, 314.269 | 320.212 |
| 2 | projection all | 52.129, 52.736, 54.293, 54.788, 55.131 | 54.293 |
| 2 | projection last | 17.705, 17.923, 17.924, 18.630, 18.088 | 17.924 |
| 4 | session creation | 22.829, 22.607, 22.032, 22.213, 22.359 | 22.359 |
| 4 | prefill | 58.569, 59.562, 58.422, 57.819, 57.776 | 58.422 |
| 4 | single decode | 19.239, 19.338, 18.678, 19.242, 19.573 | 19.242 |
| 4 | full-prefix generation | 2948.905, 2931.656, 2976.838, 3017.141, 2997.877 | 2976.838 |
| 4 | cached generation | 178.964, 178.181, 181.109, 184.129, 188.429 | 181.109 |
| 4 | projection all | 51.743, 52.594, 52.013, 52.943, 53.096 | 52.594 |
| 4 | projection last | 17.317, 17.261, 17.723, 17.090, 17.377 | 17.317 |
| 8 | session creation | 22.355, 22.080, 22.311, 22.109, 21.895 | 22.109 |
| 8 | prefill | 42.813, 44.298, 44.414, 42.888, 44.067 | 44.067 |
| 8 | single decode | 14.550, 13.921, 14.481, 14.183, 14.426 | 14.426 |
| 8 | full-prefix generation | 2932.677, 2978.027, 3008.935, 2949.889, 2971.299 | 2971.299 |
| 8 | cached generation | 146.541, 142.619, 145.572, 141.527, 143.352 | 143.352 |
| 8 | projection all | 52.119, 52.634, 52.100, 53.675, 53.141 | 52.634 |
| 8 | projection last | 18.990, 17.458, 17.202, 18.217, 18.551 | 18.217 |
| 16 | session creation | 22.572, 22.248, 21.874, 22.362, 22.067 | 22.248 |
| 16 | prefill | 53.073, 51.486, 52.530, 49.523, 52.187 | 52.187 |
| 16 | single decode | 16.368, 16.350, 16.628, 16.852, 17.021 | 16.628 |
| 16 | full-prefix generation | 3022.584, 3002.763, 3023.350, 3046.065, 2934.680 | 3022.584 |
| 16 | cached generation | 169.809, 165.722, 167.594, 167.706, 166.297 | 167.594 |
| 16 | projection all | 51.146, 52.036, 51.827, 52.406, 52.677 | 52.036 |
| 16 | projection last | 17.687, 17.612, 17.639, 17.892, 17.143 | 17.639 |

Model-load single samples for threads 1/2/4/8/16 were respectively
2408.134, 2130.635, 2441.590, 2034.827, and 2055.639 ms; they are not
generation samples. All thread counts generated the same sequence:
`15496,11,616,1438,318,1757,13,314,1101,257,6260`.

The same-build 1-to-8 comparisons are 3.862× for prefill, 3.962× for one
decode, and 4.181× for cached generation. Compared with the pre-change M2
scalar cached median, 8 threads gives 574.306/143.352 = 4.006×. The latter
uses separate sequential runs and is subject to drift: the M3 build's own
one-thread median was 599.313 ms. Sixteen threads were 16.9% slower than eight
for cached generation. The full-prefix path was not parallelized; its timed
samples are provided as a correctness/reference workload, not as M3's gain.

After serializing the scalar fallback under the same submission lock, the
primary full benchmark command was rerun at 1 and 8 threads on the final
Release source. Raw samples (ms) were:

| Threads | Metric | Five samples | Median |
|---:|---|---|---:|
| 1 | session creation | 22.639, 22.363, 23.176, 22.744, 24.088 | 22.744 |
| 1 | prefill | 181.164, 180.654, 189.586, 180.825, 181.032 | 181.032 |
| 1 | single decode | 59.089, 59.517, 60.884, 61.248, 57.140 | 59.517 |
| 1 | full-prefix generation | 2917.141, 2944.386, 2904.191, 2941.871, 2916.491 | 2917.141 |
| 1 | cached generation | 569.890, 568.260, 576.633, 583.708, 586.545 | 576.633 |
| 1 | projection all | 56.336, 55.515, 55.709, 53.484, 57.038 | 55.709 |
| 1 | projection last | 17.501, 18.256, 17.892, 18.346, 18.536 | 18.256 |
| 8 | session creation | 21.992, 22.178, 21.902, 22.595, 32.333 | 22.178 |
| 8 | prefill | 49.692, 45.351, 43.635, 43.457, 44.511 | 44.511 |
| 8 | single decode | 14.356, 14.380, 14.201, 15.456, 15.074 | 14.380 |
| 8 | full-prefix generation | 2963.522, 3003.454, 2924.977, 2996.486, 3001.476 | 2996.486 |
| 8 | cached generation | 148.227, 146.304, 143.869, 142.804, 146.945 | 146.304 |
| 8 | projection all | 52.721, 54.465, 54.845, 55.494, 55.270 | 54.845 |
| 8 | projection last | 18.229, 18.310, 17.759, 17.890, 19.534 | 18.229 |

Single model-load samples were 2287.614 and 1866.137 ms, respectively.
The final-build cached-generation ratio is 576.633/146.304 = 3.941× at
1 versus 8 threads, or 574.306/146.304 = 3.925× versus the pre-change M2
baseline. Both are consistent with the earlier sweep and do not indicate a
material performance change from the lock fix. The 8-thread final-build
prefill/decode medians were 44.511/14.380 ms. The generated token sequence
remained unchanged.

## Longer prompt: P=32, G=8

For this secondary workload the fixed 32-token prompt was
`15496,11,616,1438,318,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126`.
The command used the same benchmark with `--tokens` set to that sequence,
`--generation-tokens 8 --capacity 1024 --warmup 1 --repetitions 3`, and
`--skip-full-prefix`; it was run once with `--threads 1` and once with
`--threads 8`. Skipping the M1 timing avoided an expensive second full-prefix
workload; numeric M1 checks were performed separately.

| Threads | Metric | Three samples, ms | Median, ms |
|---:|---|---|---:|
| 1 | session creation | 22.850, 22.670, 22.186 | 22.670 |
| 1 | prefill | 1886.220, 1836.942, 1850.414 | 1850.414 |
| 1 | single decode | 62.801, 58.036, 60.259 | 60.259 |
| 1 | cached generation | 2311.691, 2296.052, 2273.712 | 2296.052 |
| 1 | projection all | 571.638, 580.762, 573.004 | 573.004 |
| 1 | projection last | 19.176, 17.950, 18.072 | 18.072 |
| 8 | session creation | 22.999, 22.395, 22.672 | 22.672 |
| 8 | prefill | 470.954, 462.895, 458.099 | 462.895 |
| 8 | single decode | 14.796, 14.193, 13.993 | 14.193 |
| 8 | cached generation | 544.420, 548.513, 559.734 | 548.513 |
| 8 | projection all | 564.715, 560.604, 557.707 | 560.604 |
| 8 | projection last | 16.903, 17.364, 17.055 | 17.055 |

Model-load single samples were 2514.788 ms (one thread) and 2537.969 ms
(eight threads). The same generated suffix was `123,126,123,126,123,126,123,126`.
At eight threads, the observed ratios were 3.997× prefill, 4.246× single
decode, and 4.186× cached generation. This is still token-by-token prefill:
M3 parallelizes each row's linear operations, not a multi-row matmul.

## Numerical checks and limitations

The llm.c checkout was verified at pinned commit
`f1e2ace651495b74ae22d45d1723443fd00ecd3a` and compiled with GCC
`-O3 -std=c11 -D_DEFAULT_SOURCE -ffp-contract=off`, with the thin harness at
`validation/llmc_reference_logits.c`. For fixed prefix `15496,11,616`, the M1
full-prefix last-position vector and upstream llm.c were all 50,257 values
byte-identical: maximum absolute/relative errors 0, argmax 1438. The command
used `validation/compare_logits.py --expected-count 50257
--absolute-tolerance 1e-4 --relative-tolerance 0 --require-bit-identical` and
POSIX `cmp`, both passing. Eight-thread cached logits for this prefix were
also byte-identical to full-prefix logits. The real-checkpoint CTest compares
four-thread cached prefill and decode against full-prefix outputs at lengths
1, 3, 4, and 5, also bit-identically. These observations do not assert an
independent upstream oracle for every prefix.

To reproduce the fixed-oracle check, set `LLMC_DIR` to a checkout of the
pinned `llm.c` commit and `CHECKPOINT` to a compatible version-1 checkpoint.
The harness includes upstream `train_gpt2.c` from that checkout:

```bash
export LLMC_DIR=/path/to/llm.c
export CHECKPOINT=/path/to/gpt2_124M.bin
git -C "$LLMC_DIR" rev-parse HEAD
gcc -O3 -std=c11 -D_DEFAULT_SOURCE -ffp-contract=off \
  -Wno-unknown-pragmas -I "$LLMC_DIR" \
  validation/llmc_reference_logits.c -lm -o /tmp/gpt2-m3-llmc-oracle
./build-m3-release/gpt2_generate --checkpoint "$CHECKPOINT" \
  --tokens 15496,11,616 --max-new-tokens 0 --greedy --mode full \
  --dump-last-logits /tmp/gpt2-m3-full.bin
./build-m3-release/gpt2_generate --checkpoint "$CHECKPOINT" \
  --tokens 15496,11,616 --max-new-tokens 0 --greedy --mode cached \
  --capacity 3 --threads 8 --dump-last-logits /tmp/gpt2-m3-cached.bin
/tmp/gpt2-m3-llmc-oracle "$CHECKPOINT" /tmp/gpt2-m3-llmc.bin
python3 validation/compare_logits.py /tmp/gpt2-m3-full.bin \
  /tmp/gpt2-m3-llmc.bin --expected-count 50257 \
  --absolute-tolerance 1e-4 --relative-tolerance 0 --require-bit-identical
cmp /tmp/gpt2-m3-full.bin /tmp/gpt2-m3-llmc.bin
python3 validation/compare_logits.py /tmp/gpt2-m3-cached.bin \
  /tmp/gpt2-m3-full.bin --expected-count 50257 \
  --absolute-tolerance 1e-4 --relative-tolerance 0 --require-bit-identical
cmp /tmp/gpt2-m3-cached.bin /tmp/gpt2-m3-full.bin
```

The scalar per-output reduction order is unchanged, but floating-point byte
identity remains a tested property of these cases, not a promise across
compilers or CPUs. The executor's fixed 65,536-floats-per-thread threshold is
empirical and can lead to nonmonotonic scaling. Pool construction and teardown
have costs outside the timed generation interval. Session calls remain
synchronous and nonconcurrent; the executor serializes simultaneous linear
submissions but does not make sharing one session concurrently safe. Neither
memory-bandwidth saturation nor a general thread-count optimum was measured.

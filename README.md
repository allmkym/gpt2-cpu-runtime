# GPT-2 CPU Inference Runtime

English | [简体中文](README.zh-CN.md)

[![CI](https://github.com/allmkym/gpt2-cpu-runtime/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/allmkym/gpt2-cpu-runtime/actions/workflows/ci.yml)

A C++20, FP32 inference runtime for GPT-2 checkpoints on a single CPU machine. It provides a full-prefix correctness baseline, an incremental KV-cache path, and an optional persistent worker executor for single-row linear layers. Input and output are token IDs; the repository does not include model weights or a tokenizer.

## Project origin

This repository is a C++20 follow-on runtime derived from the GPT-2 CPU inference implementation in [karpathy/llm.c](https://github.com/karpathy/llm.c/tree/f1e2ace651495b74ae22d45d1723443fd00ecd3a), following the inference-focused subset used in JYY OS 2026 M6, which I previously completed as a C systems/concurrency exercise. It does not contain my original M6 submission or the course framework. The runtime adds explicit C++ ownership and lifetime boundaries, incremental KV-cache inference, a persistent linear executor, and additional correctness and performance validation.

## Engineering highlights

- **Explicit ownership:** `ModelWeights` owns one contiguous parameter buffer; `InferenceWorkspace` and `InferenceSession` own their mutable state. Tensor and logits views use `std::span` with documented lifetimes.
- **Incremental inference:** a fixed-capacity session stores K/V by layer and position. `prefill` consumes a prompt token by token; `decode` processes one additional token without recomputing previous tokens.
- **Bounded parallelism:** `LinearExecutor` keeps `N-1` workers for `N` total threads, with the caller computing one output shard. Disjoint output-channel ranges retain each channel's scalar accumulation order. Small jobs use the scalar kernel.
- **Numerical checks:** synthetic tests, real-checkpoint comparisons, and an independent pinned `llm.c` oracle compare complete logits vectors, rather than only generated token IDs.

## Measured results

On an Intel Core i7-14700HX under WSL2 with GCC 15.2 Release, GPT-2 124M, a 3-token prompt and 8 generated tokens, the recorded M2/M3 measurements include:

| Experiment | Baseline median | Optimized median | Ratio |
|---|---:|---:|---:|
| Full-prefix → scalar cached generation (M2) | 2924.550 ms | 583.515 ms | 5.012× |
| Cached generation, 1 → 8 total threads (final M3) | 576.633 ms | 146.304 ms | 3.941× |

These are two distinct experiments and should not be multiplied or treated as a factorial comparison. The M2 result reflects the combined cached path (KV reuse, one-token execution, and last-position real-vocabulary projection); the M3 result is a same-build 1-vs-8-thread comparison for the optional parallel linear executor. Each median used one warmup and five timed repetitions. Model loading, session creation, executor construction, logging, and logits dumps were outside generation timing. These are bounded local measurements, not portable thread-count recommendations. Raw samples and additional workloads are in [implementation notes](IMPLEMENTATION_NOTES.md) and [M3 results](docs/MILESTONE3_RESULTS.md).

## Build and run

Requirements: CMake 3.20+, a C++20 compiler with thread support, and Python 3.10+ when tests are enabled. The recorded builds used GCC on Ubuntu under WSL2.

```bash
cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release
cmake --build build-release --parallel
ctest --test-dir build-release --output-on-failure
```

The default configuration runs synthetic and CLI tests without a checkpoint. To include the real-model integration test, supply a compatible GPT-2 checkpoint when configuring:

```bash
export CHECKPOINT=/path/to/gpt2_124M.bin
cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release \
  -DGPT2_TEST_CHECKPOINT="$CHECKPOINT"
cmake --build build-release --parallel
ctest --test-dir build-release -N
ctest --test-dir build-release --output-on-failure
```

The test list should include `gpt2_real_checkpoint` when the file exists. The recorded final-source Release, ASan/LSan/UBSan, and separate TSan suites each passed 6/6 with that checkpoint. Sanitizer runs are documented in [M3 results](docs/MILESTONE3_RESULTS.md); they are separate from the Release timing build.

Generate two tokens with the full-prefix baseline:

```bash
./build-release/gpt2_generate --checkpoint "$CHECKPOINT" \
  --tokens 15496,11,616 --max-new-tokens 2 --greedy --mode full
```

Use the KV cache and optional parallel linear executor:

```bash
./build-release/gpt2_generate --checkpoint "$CHECKPOINT" \
  --tokens 15496,11,616 --max-new-tokens 2 --greedy \
  --mode cached --capacity 1024 --threads 8
```

Both commands produced `15496 11 616 1438 318` with the recorded GPT-2 124M checkpoint. `--threads` includes the calling thread and is available only in cached mode; its default is 1. Full mode remains the independent scalar path.

## Design and correctness

`GPT2Model::forward` recomputes the complete input prefix and returns a view into caller-owned workspace. `InferenceSession` borrows an existing model and owns the KV cache, one-token scratch, and last-position real-vocabulary logits. A model must outlive its sessions and remain unmoved while they exist. A session does not support concurrent calls; its returned logits view is overwritten by subsequent mutating calls.

The session's cache layout is `[layer][position][channel]` in separate contiguous K and V vectors. Capacity is fixed at construction. `prefill` validates the whole prompt before computation and publishes the consumed length only after success; `decode` appends at the next available position. Each token's Q attends only to K/V in its visible prefix, including itself.

The parallel executor is borrowed only during a synchronous `prefill` or `decode` call. Each linear job partitions output channels into disjoint ranges and waits for every worker before returning or propagating an exception. The full-prefix path, attention, LayerNorm, GELU, and token-by-token prefill are unchanged by the executor.

For the fixed prefix `15496,11,616`, the full-prefix logits were byte-identical to the pinned upstream `llm.c` oracle across all 50,257 real-vocabulary values; eight-thread cached logits were byte-identical to the full-prefix logits. A separate real-checkpoint test compared four-thread cached and full-prefix logits at prefix lengths 1, 3, 4, and 5. See [implementation notes](IMPLEMENTATION_NOTES.md) and the [M2](docs/MILESTONE2_DESIGN.md) and [M3](docs/MILESTONE3_DESIGN.md) design records for the precise contracts and validation boundaries.

## Benchmark commands

```bash
./build-release/gpt2_benchmark --checkpoint "$CHECKPOINT" \
  --tokens 15496,11,616 --generation-tokens 8 --capacity 1024 \
  --warmup 1 --repetitions 5 --threads 8

./build-release/gpt2_linear_benchmark "$CHECKPOINT" 1 5 8 8
```

The first command reports full-prefix and cached generation, prefill, one decode, and separately timed setup and projection probes. The second measures single-row linear shapes; its final argument is total thread count. Compare thread counts using the same build and workload.

## Scope and references

The runtime is single-sequence and synchronous, with FP32 computation, greedy token selection, fixed cache capacity, and no text tokenization. It does not implement training, batching, quantization, BLAS integration, multi-row prefill, or a serving API.

The checkpoint conventions, full-prefix computation, and independent numerical reference trace to [karpathy/llm.c](https://github.com/karpathy/llm.c/tree/f1e2ace651495b74ae22d45d1723443fd00ecd3a). The incremental-state design also references [llama2.c](https://github.com/karpathy/llama2.c/tree/350e04fe35433e6d2941dce5a1f53308f87058eb) and [ggml](https://github.com/ggml-org/ggml/tree/7840aaba1989c6deeefede1d77d5aaf8f52b947e). Detailed source and license notes are collected in [implementation notes](IMPLEMENTATION_NOTES.md) and [third-party notices](THIRD_PARTY_NOTICES.md).

## License

This project is released under the [MIT License](LICENSE). Portions adapted from `karpathy/llm.c` retain the upstream copyright and MIT notice; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

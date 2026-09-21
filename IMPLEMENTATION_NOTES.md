# Implementation and Validation Notes

This document records the runtime's principal design decisions and the observed validation results. Build and usage instructions are in [README.md](README.md); M2 and M3 mechanisms and full M3 raw samples are in [docs/MILESTONE2_DESIGN.md](docs/MILESTONE2_DESIGN.md), [docs/MILESTONE3_DESIGN.md](docs/MILESTONE3_DESIGN.md), and [docs/MILESTONE3_RESULTS.md](docs/MILESTONE3_RESULTS.md).

## Runtime structure

| Component | Responsibility |
|---|---|
| `ModelWeights` / `ParameterLayout` | Validate a checkpoint and own one contiguous parameter allocation with checked tensor offsets. |
| `GPT2Model::forward` / `InferenceWorkspace` | Run a scalar full-prefix forward pass using caller-owned activation storage. |
| `InferenceSession` / `KVCache` | Maintain one sequence's fixed-capacity K/V cache, one-token scratch, and last-position logits. |
| `LinearExecutor` | Run an optional synchronous single-row linear job across persistent worker threads and the submitting thread. |
| `gpt2_generate` | Expose full and cached greedy token-ID generation. |
| `gpt2_benchmark` / `gpt2_linear_benchmark` | Measure complete generation stages and individual linear shapes. |

The forward path performs token and position embeddings, `L` pre-norm Transformer blocks, a final LayerNorm, and a tied token-embedding projection. Weight matrices use output-major rows. Full-prefix inference processes every position; cached inference uses the same one-token core for prefill and decode. For a newly consumed token at position `p`, the session writes that layer's K/V before its Q attends to positions `[0,p]`.

## Ownership and failure boundaries

- `ModelWeights` owns the parameter vector, cannot be copied, and can be moved. `TensorView` and `std::span` borrow data; they do not extend the owner's lifetime.
- `InferenceWorkspace` owns full-prefix activations and logits. `InferenceSession` owns KV cache, scratch, logits, and its published consumed-token count while borrowing a stable `GPT2Model`.
- The same session cannot be called concurrently. A returned logits span is a view of reusable storage, so callers must copy it to preserve historical results.
- `LinearExecutor` owns `N-1` worker `std::thread` objects for `N` total threads. Each job borrows input, weights, bias, and output only until `linear()` returns. A submission lock serializes simultaneous calls, including scalar fallback; a separate state lock and condition variables coordinate publication and completion. Destruction stops, wakes, and joins workers.
- The checkpoint loader checks header size, magic, version, positive dimensions, shape compatibility, arithmetic overflow, exact payload length, and read completion before producing a model. It supports checkpoint versions 1 and 3, including padded vocabulary in version 3.
- Session methods validate expected caller errors before publishing a new logical length. Worker exceptions are captured and returned through the submitting thread after all shards finish, preserving the borrowed-data lifetime boundary.

## Recorded validation

| Stage | Checks | Observed result |
|---|---|---|
| Full-prefix baseline | Synthetic loader/kernel tests, real GPT-2 124M forward, pinned upstream `llm.c` last-position logits. | All 50,257 real-vocabulary logits were byte-identical for prefix `15496,11,616`. |
| Cached inference | Cache indexing and visibility, state and capacity errors, synthetic nonzero checkpoints, CLI boundaries, real-checkpoint full/cached parity. | Release and ASan/LSan/UBSan CTest each passed 5/5. Real-checkpoint logits matched at prefix lengths 1, 3, 4, and 5. |
| Parallel linear | Scalar/parallel output parity, uneven shards, bias/no-bias, concurrent submissions, scalar fallback, real-checkpoint cached/full parity. | Release, ASan/LSan/UBSan with `_GLIBCXX_ASSERTIONS`, and separate TSan CTest each passed 6/6. Fixed-prefix 8-thread cached logits matched full-prefix logits byte for byte. |

The pinned `llm.c` commit is `f1e2ace651495b74ae22d45d1723443fd00ecd3a`. Both oracle and C++ builds used `-ffp-contract=off` for the recorded byte-level comparison. The oracle covers the fixed prefix; the stepwise full/cached comparisons are internal path comparisons, not separate upstream oracle runs for every prefix. The sanitizer results describe the paths exercised by those tests and are not a proof of absence of every defect.

The final M3 executor tests cover 0-thread rejection, total thread counts 1/2/4/8, odd output sizes, both bias modes, size-error recovery, concurrent parallel submissions, concurrent scalar fallback, and repeated construction/destruction. They do not specifically inject a worker computation exception or simulate a partially failed thread constructor. Those error-handling paths are implemented but should not be described as directly exercised by the existing tests.

## Recorded performance

The following measurements were made on an Intel Core i7-14700HX under WSL2 with GCC 15.2 Release, GPT-2 124M, prompt `15496,11,616`, eight generated tokens, capacity 1024, one warmup, and five samples. Model loading, session creation, and log or logits-dump work are outside generation timings.

| Comparison | Median times | Ratio | Interpretation |
|---|---|---:|---|
| Full-prefix vs scalar cached, earlier M2 build | 2924.550 vs 583.515 ms | 5.012× | Combined effect of KV reuse, one-token execution, and last-position real-vocabulary projection. |
| Cached 1 vs 8 total threads, final M3 build | 576.633 vs 146.304 ms | 3.941× | Same-build comparison for optional parallel linear work. |

These are distinct experiments and should not be combined as if they were one factorial benchmark. See [M3 results](docs/MILESTONE3_RESULTS.md) for the thread-count sweep, raw samples, a longer-prompt workload, per-shape measurements, commands, and complete environment details.

## Scope

The runtime uses FP32, batch size 1, token-ID I/O, greedy decoding, fixed cache capacity, and synchronous calls. Prefill remains token by token and computes intermediate per-token logits; attention, LayerNorm, GELU, and the full-prefix oracle remain scalar. It does not include training, tokenizer/text input, batching, quantization, BLAS, a general task scheduler, or a serving framework. Performance varies with CPU, compiler, checkpoint, input length, machine load, and thread count.

## External references

- GPT-2 checkpoint conventions and independent CPU numerical oracle: [karpathy/llm.c at the pinned commit](https://github.com/karpathy/llm.c/tree/f1e2ace651495b74ae22d45d1723443fd00ecd3a) and its [MIT license](https://github.com/karpathy/llm.c/blob/f1e2ace651495b74ae22d45d1723443fd00ecd3a/LICENSE).
- Course exercise context and inference-focused subset of `llm.c`: [JYY OS 2026 M6](https://git.nju.edu.cn/jyy/os2026), `M6:gpt/gpt.c`.
- Fixed-capacity incremental-state design reference: [karpathy/llama2.c](https://github.com/karpathy/llama2.c/tree/350e04fe35433e6d2941dce5a1f53308f87058eb) and its [MIT license](https://github.com/karpathy/llama2.c/blob/350e04fe35433e6d2941dce5a1f53308f87058eb/LICENSE).
- GPT-2 past-positioning design reference: [ggml](https://github.com/ggml-org/ggml/tree/7840aaba1989c6deeefede1d77d5aaf8f52b947e) and its [MIT license](https://github.com/ggml-org/ggml/blob/7840aaba1989c6deeefede1d77d5aaf8f52b947e/LICENSE).

Model weights and upstream source trees are not included in this repository.

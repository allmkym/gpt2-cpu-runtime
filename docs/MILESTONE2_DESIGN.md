# Milestone 2 Design: Single-Sequence Incremental Inference

## Incremental path

The cached runtime adds a reusable, serial FP32 incremental path for one
GPT-2 model and one token sequence. It eliminates generation-time recomputation
of already-consumed tokens while retaining `GPT2Model::forward` and
`InferenceWorkspace` unchanged as the independent full-prefix oracle.

This path handles batch size 1, a fixed-capacity KV cache, token-by-token
prefill, one-token decode, reset, reusable scratch, and logits for the real
vocabulary at the current last position. The session also accepts the optional
single-row linear executor described in [M3 design](MILESTONE3_DESIGN.md);
the cache layout and state machine below remain the same. SIMD intrinsics,
BLAS, quantization, tokenization, multiple requests, and serving are outside
the current runtime scope.

## Objects and ownership

```text
GPT2Model (owns immutable ModelWeights)
  |
  +-- borrowed by InferenceSession
        +-- owns KVCache (one contiguous K vector and one contiguous V vector)
        +-- owns fixed-size one-token scratch
        +-- owns real-vocabulary last-position logits
        +-- publishes the number of successfully consumed tokens
```

`InferenceSession` borrows a `const GPT2Model`; it neither owns the model nor
uses shared ownership. The model must outlive the session and must not be moved
while the session exists. Construction from an rvalue model is deleted.
Sessions are neither copyable nor movable, and the same session must not be
called concurrently.

Returned logits are a view of session-owned memory and contain exactly
`vocabulary_size` values, never padded-vocabulary entries. A subsequent
`prefill`, `decode`, `reset`, or destruction invalidates the result as a value:
the address may remain stable while the contents are overwritten. A caller
that needs historical logits must copy them.

## KV cache layout and allocation

K and V use separate contiguous `std::vector<float>` allocations:

```text
offset(layer, position, channel) =
    (layer * capacity + position) * channels + channel

K[layer][position][channel]
V[layer][position][channel]
```

`channel` is conceptually `[head][head_dim]`. This layout keeps a token's
channels contiguous and makes cache writes and per-head dot products explicit.
It is a simple correctness-first layout, not a claim of optimal CPU locality.

Session capacity is fixed at construction and must be in
`[1, model.max_sequence_length]`. All products and allocation sizes are checked
before allocation. Decode never grows a vector or creates per-layer temporary
vectors. Reset changes only logical length; cache bytes need not be cleared,
because every attention call is bounded by the newly published prefix length.

For L layers, S positions, and C channels, cache storage is
`2 * L * S * C * sizeof(float)` bytes. GPT-2 124M at capacity 1024 therefore
uses 72 MiB for K and V.

## State and failure semantics

The public state is intentionally only Empty (`size() == 0`) or Ready
(`size() > 0`).

| Operation | Preconditions | Successful result |
|---|---|---|
| `prefill(prompt)` | Empty; prompt nonempty; all IDs valid; prompt fits | Consumes the whole prompt and returns its next-token logits |
| `decode(token)` | Ready; ID valid; one cache slot remains | Appends at the old `size()` and returns new next-token logits |
| `reset()` | No concurrent call | Returns to Empty while retaining allocations |

`prefill` validates the complete prompt before any cache write. Expected input
errors are rejected before changing effective state. Internal computation uses
an unpublished working position; `size_` changes only after the complete public
operation succeeds. Writes left by an unexpected exception are unreachable
while `size_` remains unchanged and are overwritten by later valid work.

## Incremental execution

Prefill deliberately loops over the same private single-token core used by
decode. For a token at absolute position `p`, that core:

1. adds token and learned absolute-position embeddings;
2. for each layer, runs one-row LN and QKV projection;
3. stores that layer's current K and V at position `p`;
4. computes attention for the current Q over cache positions `[0, p]`, including
   itself, with a stable softmax;
5. runs the attention projection, residual, second LN, MLP, and residual;
6. runs final LN and a tied-embedding projection for only the real vocabulary.

The cached attention kernel is separate from M1
`causal_self_attention`; the reference path is not routed through new cache
control flow. With scalar kernels and the same per-output accumulation order,
the target is bit-identical real-vocabulary logits on the pinned toolchain.

## Generation timeline

After prefill, logits already select the first generated token. Generating G
tokens normally requires G-1 decode calls:

```text
prefill(prompt) -> choose generated[0]
decode(generated[0]) -> choose generated[1]
...
decode(generated[G-2]) -> choose generated[G-1]
```

Thus the final generated token normally has not been consumed, and output
length may exceed session size by one. `--max-new-tokens 0` consumes only the
prompt and emits it unchanged. If `--dump-last-logits` requests logits for the
complete final output sequence, cached generation performs one extra decode of
the final generated token; benchmark timing excludes this optional dump work.

## Verification and measurement

Tests cover cache capacity/indexing/overflow, future-region poisoning, session
state and lifetime traits, deterministic nonzero v1/v3 checkpoints (including
padded vocabulary), two interleaved sessions, reset reuse, CLI generation
boundaries, real-checkpoint stepwise comparison, finite logits, and known greedy
tokens. M1 unit, integration, and pinned llm.c oracle checks remain in place.

The benchmark reports model load, session construction, prefill, one-step
decode, full-prefix generation, and cached generation separately, with warmup,
raw repeated samples, and medians. A projection microbenchmark compares T-row
versus one-row tied-embedding work to identify output-projection clipping; the
end-to-end full-versus-cached comparison includes cache reuse. Release results
must record compiler flags, CPU, WSL/kernel, checkpoint/input, repetitions, and
thread count (one).

## Design references

The checkpoint organization and numerical oracle follow the pinned
[`llm.c`](https://github.com/karpathy/llm.c/tree/f1e2ace651495b74ae22d45d1723443fd00ecd3a)
CPU implementation. The fixed-capacity state layout was also informed by
[`llama2.c`](https://github.com/karpathy/llama2.c/tree/350e04fe35433e6d2941dce5a1f53308f87058eb)
RunState and [`ggml`](https://github.com/ggml-org/ggml/tree/7840aaba1989c6deeefede1d77d5aaf8f52b947e)
GPT-2 past-position handling. Pinned commits and license links are collected
in [implementation notes](../IMPLEMENTATION_NOTES.md).

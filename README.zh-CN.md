# GPT-2 CPU Inference Runtime

[English](README.md) | 简体中文

[![CI](https://github.com/allmkym/gpt2-cpu-runtime/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/allmkym/gpt2-cpu-runtime/actions/workflows/ci.yml)

一个面向单机 CPU、使用 C++20 与 FP32 的 GPT-2 inference runtime。项目提供 full-prefix correctness baseline、incremental KV cache path，以及用于 single-row Linear 的可选 persistent worker executor。输入和输出均为 token ID；仓库不包含 model weights 或 tokenizer。

## 项目来源

本仓库基于 [karpathy/llm.c](https://github.com/karpathy/llm.c/tree/f1e2ace651495b74ae22d45d1723443fd00ecd3a) 中的 GPT-2 CPU inference implementation 继续开发，并沿用了 JYY OS 2026 M6 所采用的 inference-focused subset；我此前已经完成该 M6 的 C systems/concurrency exercise。本仓库不包含我原始的 M6 提交或课程 framework。在此基础上，项目进一步加入了明确的 C++ ownership / lifetime boundaries、incremental KV cache inference、persistent LinearExecutor，以及额外的 correctness / performance validation。

## 工程要点

- **Ownership / lifetime：** `ModelWeights` 持有一块连续的 parameter buffer；`InferenceWorkspace` 与 `InferenceSession` 持有各自的 mutable state。Tensor 与 logits view 使用 `std::span`，并明确约束其 lifetime。
- **Incremental inference：** 固定容量的 session 按 layer 与 position 保存 K/V。`prefill` 按 token 逐个处理 prompt；`decode` 每次仅处理一个新增 token，不重新计算此前 token。
- **Bounded parallelism：** 当 total thread count 为 `N` 时，`LinearExecutor` 常驻 `N-1` 个 worker，caller 自身负责一个 output shard。各线程处理互不重叠的 output-channel ranges，因此每个 output channel 内部仍保持与 scalar implementation 相同的 accumulation order；较小的 job 会回退到 scalar kernel。
- **Numerical validation：** 使用 synthetic tests、real-checkpoint comparisons，以及固定版本的独立 `llm.c` oracle；比较的是完整 logits vectors，而不仅是最终生成的 token ID。

## 实测结果

在 Intel Core i7-14700HX、WSL2、GCC 15.2 Release 环境下，使用 GPT-2 124M、3-token prompt 和 8 个 generated tokens，记录到的 M2/M3 结果包括：

| Experiment | Baseline median | Optimized median | Ratio |
|---|---:|---:|---:|
| Full-prefix → scalar cached generation（M2） | 2924.550 ms | 583.515 ms | 5.012× |
| Cached generation，1 → 8 total threads（final M3） | 576.633 ms | 146.304 ms | 3.941× |

这两组数据来自不同实验，不能相乘，也不能视为同一个 factorial comparison。M2 的提升来自整条 cached path 的综合变化，包括 KV reuse、one-token execution 和 last-position real-vocabulary projection；M3 则是在同一 build 下，对可选 parallel `LinearExecutor` 进行 1-thread 与 8-total-thread 对比。每组 median 均采用 1 次 warmup 和 5 次 timed repetitions。Model loading、session creation、executor construction、logging 和 logits dump 均不计入 generation timing。

这些结果只代表这台机器和该 workload 下的 bounded local measurements，不应视为 portable thread-count recommendation。Raw samples 和额外 workload 见 [实现与验证说明](IMPLEMENTATION_NOTES.md) 与 [M3 结果记录](docs/MILESTONE3_RESULTS.md)。

## 构建与运行

要求：CMake 3.20+、支持 threads 的 C++20 compiler；启用测试时需要 Python 3.10+。记录中的 build environment 为 WSL2 下的 Ubuntu + GCC。

```bash
cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release
cmake --build build-release --parallel
ctest --test-dir build-release --output-on-failure
```

默认配置会运行不依赖 checkpoint 的 synthetic tests 与 CLI tests。如需运行 real-model integration test，在配置阶段提供兼容的 GPT-2 checkpoint：

```bash
export CHECKPOINT=/path/to/gpt2_124M.bin
cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release \
  -DGPT2_TEST_CHECKPOINT="$CHECKPOINT"
cmake --build build-release --parallel
ctest --test-dir build-release -N
ctest --test-dir build-release --output-on-failure
```

当 checkpoint 文件存在时，测试列表中应包含 `gpt2_real_checkpoint`。使用该 checkpoint 时，记录中的 final-source Release、ASan/LSan/UBSan，以及单独的 TSan suites 均通过 6/6。Sanitizer runs 见 [M3 结果记录](docs/MILESTONE3_RESULTS.md)；这些 runs 与 Release timing build 相互独立。

使用 full-prefix baseline 生成两个 token：

```bash
./build-release/gpt2_generate --checkpoint "$CHECKPOINT" \
  --tokens 15496,11,616 --max-new-tokens 2 --greedy --mode full
```

使用 KV cache 和可选 parallel LinearExecutor：

```bash
./build-release/gpt2_generate --checkpoint "$CHECKPOINT" \
  --tokens 15496,11,616 --max-new-tokens 2 --greedy \
  --mode cached --capacity 1024 --threads 8
```

在记录使用的 GPT-2 124M checkpoint 上，两条命令都会生成 `15496 11 616 1438 318`。`--threads` 表示包含 caller 在内的 total thread count，只在 cached mode 下可用，默认值为 1；full mode 始终保持独立的 scalar path。

## Design / Correctness

`GPT2Model::forward` 会重新计算完整 input prefix，并返回指向 caller-owned workspace 的 view。`InferenceSession` 借用已有 model，同时自行持有 KV cache、one-token scratch 和 last-position real-vocabulary logits。model 的 lifetime 必须覆盖所有 session，且 session 存活期间不能移动 model。单个 session 不支持 concurrent calls；后续 mutating call 会覆盖此前返回的 logits view。

session 的 cache layout 为独立连续 K/V vectors 中的 `[layer][position][channel]`，capacity 在 construction 时固定。`prefill` 会先校验完整 prompt，再开始计算，并且只有成功后才更新 consumed length；`decode` 会在下一个 available position 追加 token。每个 token 的 Q 只对 visible prefix 中的 K/V 做 attention，并包含当前位置自身。

parallel executor 只会在同步的 `prefill` 或 `decode` call 期间被借用。每个 Linear job 按 output channels 划分为 disjoint ranges，并在返回或传播 exception 之前等待所有 worker 完成。full-prefix path、attention、LayerNorm、GELU，以及 token-by-token prefill 不会因为 executor 而改变。

对于固定 prefix `15496,11,616`，full-prefix logits 在全部 50,257 个 real-vocabulary values 上与固定版本的 upstream `llm.c` oracle byte-identical；8-thread cached logits 与 full-prefix logits 也 byte-identical。另一个 real-checkpoint test 在 prefix lengths 1、3、4、5 处比较了 4-thread cached 与 full-prefix logits。更精确的 contracts 与 validation boundaries 见 [实现与验证说明](IMPLEMENTATION_NOTES.md)、[M2 设计记录](docs/MILESTONE2_DESIGN.md) 和 [M3 设计记录](docs/MILESTONE3_DESIGN.md)。

## Benchmark commands

```bash
./build-release/gpt2_benchmark --checkpoint "$CHECKPOINT" \
  --tokens 15496,11,616 --generation-tokens 8 --capacity 1024 \
  --warmup 1 --repetitions 5 --threads 8

./build-release/gpt2_linear_benchmark "$CHECKPOINT" 1 5 8 8
```

第一个命令会报告 full-prefix / cached generation、prefill、one decode，以及单独计时的 setup / projection probes。第二个命令测试 single-row Linear shapes；最后一个参数表示 total thread count。比较不同 thread counts 时应保持相同 build 和 workload。

## Scope / References

当前 runtime 为 single-sequence、synchronous，采用 FP32 computation、greedy token selection 和 fixed cache capacity，不提供 text tokenizer。项目不实现 training、batching、quantization、BLAS integration、multi-row prefill 或 serving API。

Checkpoint conventions、full-prefix computation 与 independent numerical reference 均可追溯到 [karpathy/llm.c](https://github.com/karpathy/llm.c/tree/f1e2ace651495b74ae22d45d1723443fd00ecd3a)。Incremental-state design 还参考了 [llama2.c](https://github.com/karpathy/llama2.c/tree/350e04fe35433e6d2941dce5a1f53308f87058eb) 和 [ggml](https://github.com/ggml-org/ggml/tree/7840aaba1989c6deeefede1d77d5aaf8f52b947e)。更详细的 provenance / license 说明见 [实现与验证说明](IMPLEMENTATION_NOTES.md) 与 [第三方声明](THIRD_PARTY_NOTICES.md)。

## License

本项目采用 [MIT License](LICENSE)。由 `karpathy/llm.c` adapted 的部分保留 upstream copyright 与 MIT notice，详见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

> 更深入的 design、validation 和 raw benchmark records 继续维护单一英文版本，以避免中英文证据文档发生漂移。

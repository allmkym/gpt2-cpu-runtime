# GPT-2 CPU 推理运行时

[English](README.md) | 简体中文

[![CI](https://github.com/allmkym/gpt2-cpu-runtime/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/allmkym/gpt2-cpu-runtime/actions/workflows/ci.yml)

一个面向单机 CPU、基于 C++20 和 FP32 的 GPT-2 checkpoint 推理运行时。项目包含完整前缀（full-prefix）正确性基线、增量 KV Cache 推理路径，以及针对单行 Linear 的可选持久工作线程执行器。输入和输出均为 token ID；仓库不包含模型权重或 tokenizer。

## 项目来源

本仓库是基于 [karpathy/llm.c](https://github.com/karpathy/llm.c/tree/f1e2ace651495b74ae22d45d1723443fd00ecd3a) 中 GPT-2 CPU 推理实现发展而来的 C++20 后续运行时，并沿用了 JYY OS 2026 M6 所采用的面向推理的裁剪范围；我此前将 M6 作为 C 语言系统与并发练习完成。本仓库不包含我原始的 M6 提交或课程框架。在此基础上，项目进一步加入了明确的 C++ 所有权与生命周期边界、增量 KV Cache 推理、持久化 Linear 执行器，以及额外的正确性和性能验证。

## 工程要点

- **明确的所有权与生命周期：** `ModelWeights` 持有一块连续的参数缓冲区；`InferenceWorkspace` 与 `InferenceSession` 持有各自的可变状态。Tensor 与 logits 视图使用 `std::span`，并明确约束其生命周期。
- **增量推理：** 固定容量的 session 按 layer 与 position 保存 K/V。`prefill` 按 token 逐个消费 prompt；`decode` 每次仅处理一个新增 token，不重新计算此前 token。
- **有界并行：** 当总线程数为 `N` 时，`LinearExecutor` 常驻 `N-1` 个 worker，调用线程自身负责一个输出 shard。各线程处理互不重叠的输出通道范围，因此每个输出通道内部仍保持与标量实现相同的累加顺序；较小的任务会回退到标量 kernel。
- **数值正确性验证：** 使用合成测试、真实 checkpoint 路径对比，以及固定版本的独立 `llm.c` oracle；比较的是完整 logits 向量，而不仅是最终生成的 token ID。

## 实测结果

在 Intel Core i7-14700HX、WSL2、GCC 15.2 Release 环境下，使用 GPT-2 124M、3-token prompt 和 8 个生成 token，记录到的 M2/M3 结果包括：

| 实验 | 基线中位数 | 优化后中位数 | 比值 |
|---|---:|---:|---:|
| 完整前缀 → 标量 KV Cache 生成（M2） | 2924.550 ms | 583.515 ms | 5.012× |
| KV Cache 生成，1 → 8 个总线程（最终 M3） | 576.633 ms | 146.304 ms | 3.941× |

这两组数据来自两个不同实验，不能相乘，也不能当作同一个析因实验的结果。M2 的提升来自整条 cached path 的综合变化，包括 KV 复用、单 token 执行，以及只对最后位置进行真实词表投影；M3 则是在同一最终构建中，对可选并行 Linear 执行器进行 1 线程与 8 个总线程的对比。每组中位数均采用 1 次 warmup 和 5 次计时重复。模型加载、session 创建、executor 构造、日志和 logits dump 均不计入 generation 时间。

M3 的线程数 sweep 测试了 1、2、4、8、16 个总线程；在这组 workload 上 8 线程最好，而 16 线程出现回退。因此这些结果只代表该机器与该 workload 下的本地测量，不应视为可移植的线程数推荐。原始样本和额外 workload 见 [实现与验证说明](IMPLEMENTATION_NOTES.md) 与 [M3 结果记录](docs/MILESTONE3_RESULTS.md)。

## 构建与运行

要求：CMake 3.20+、支持线程的 C++20 编译器；启用测试时需要 Python 3.10+。记录中的构建环境为 WSL2 下的 Ubuntu + GCC。

```bash
cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release
cmake --build build-release --parallel
ctest --test-dir build-release --output-on-failure
```

默认配置会运行不依赖 checkpoint 的合成测试与 CLI 测试。如需运行真实模型集成测试，在配置阶段提供兼容的 GPT-2 checkpoint：

```bash
export CHECKPOINT=/path/to/gpt2_124M.bin
cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release \
  -DGPT2_TEST_CHECKPOINT="$CHECKPOINT"
cmake --build build-release --parallel
ctest --test-dir build-release -N
ctest --test-dir build-release --output-on-failure
```

当 checkpoint 文件存在时，测试列表中应包含 `gpt2_real_checkpoint`。使用该 checkpoint 时，记录中的最终源码版本在 Release、ASan/LSan/UBSan，以及单独的 TSan 测试中均通过 6/6。Sanitizer 运行记录见 [M3 结果记录](docs/MILESTONE3_RESULTS.md)；这些测试与 Release 性能计时使用的是不同构建。

使用完整前缀基线生成两个 token：

```bash
./build-release/gpt2_generate --checkpoint "$CHECKPOINT" \
  --tokens 15496,11,616 --max-new-tokens 2 --greedy --mode full
```

使用 KV Cache 和可选的并行 Linear 执行器：

```bash
./build-release/gpt2_generate --checkpoint "$CHECKPOINT" \
  --tokens 15496,11,616 --max-new-tokens 2 --greedy \
  --mode cached --capacity 1024 --threads 8
```

在记录使用的 GPT-2 124M checkpoint 上，两条命令都会生成 `15496 11 616 1438 318`。`--threads` 表示包含调用线程在内的总线程数，只在 cached mode 下可用，默认值为 1；full mode 始终保持独立的标量路径。

## 设计与正确性

`GPT2Model::forward` 会重新计算完整输入前缀，并返回指向调用者所持有 workspace 的视图。`InferenceSession` 借用已有的 model，同时自行持有 KV Cache、单 token scratch 和最后位置的真实词表 logits。model 的生命周期必须长于其 session，并且在 session 存活期间不能被移动。单个 session 不支持并发调用；后续修改性调用会覆盖之前返回的 logits 视图内容。

session 的 cache layout 为独立连续 K/V 向量中的 `[layer][position][channel]`，容量在构造时固定。`prefill` 会先验证完整 prompt，再开始计算，并且只有成功后才对外更新已消费长度；`decode` 会在下一个可用 position 追加 token。每个 token 的 Q 只会关注其可见前缀中的 K/V，并包含当前位置自身。

并行 executor 只会在同步的 `prefill` 或 `decode` 调用期间被借用。每个 Linear job 按输出通道划分为互不重叠的区间，并在返回或传播异常之前等待所有 worker 完成。完整前缀路径、attention、LayerNorm、GELU，以及逐 token 的 prefill 语义不会因为 executor 而改变。

对于固定前缀 `15496,11,616`，完整前缀路径在全部 50,257 个真实词表 logits 上与固定版本的上游 `llm.c` oracle 实现字节级一致；8 线程 cached logits 与完整前缀 logits 也字节级一致。另一个真实 checkpoint 测试在前缀长度 1、3、4、5 处比较了 4 线程 cached 路径与完整前缀路径。更精确的契约与验证边界见 [实现与验证说明](IMPLEMENTATION_NOTES.md)、[M2 设计记录](docs/MILESTONE2_DESIGN.md) 和 [M3 设计记录](docs/MILESTONE3_DESIGN.md)。

## 性能测试命令

```bash
./build-release/gpt2_benchmark --checkpoint "$CHECKPOINT" \
  --tokens 15496,11,616 --generation-tokens 8 --capacity 1024 \
  --warmup 1 --repetitions 5 --threads 8

./build-release/gpt2_linear_benchmark "$CHECKPOINT" 1 5 8 8
```

第一个命令会报告完整前缀与 cached generation、prefill、一次 decode，以及单独计时的 setup 与 projection probe。第二个命令测试单行 Linear 的不同 shape；最后一个参数表示总线程数。比较不同线程数时应保持构建版本和 workload 一致。

## 范围与参考

当前 runtime 面向单序列同步调用，使用 FP32、greedy token selection 和固定容量 cache，不提供文本 tokenizer。项目不实现训练、batching、quantization、BLAS integration、多行 prefill 或 serving API。

checkpoint 约定、完整前缀计算以及独立数值 reference 均可追溯到 [karpathy/llm.c](https://github.com/karpathy/llm.c/tree/f1e2ace651495b74ae22d45d1723443fd00ecd3a)。增量状态设计还参考了 [llama2.c](https://github.com/karpathy/llama2.c/tree/350e04fe35433e6d2941dce5a1f53308f87058eb) 和 [ggml](https://github.com/ggml-org/ggml/tree/7840aaba1989c6deeefede1d77d5aaf8f52b947e)。更详细的来源和许可证说明见 [实现与验证说明](IMPLEMENTATION_NOTES.md) 与 [第三方声明](THIRD_PARTY_NOTICES.md)。

## 许可证

本项目采用 [MIT License](LICENSE)。由 `karpathy/llm.c` 改编的部分保留上游版权与 MIT 声明，详见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

> 更深入的设计、验证和原始 benchmark 记录继续维护单一英文版本，以避免中英文证据文档发生漂移。

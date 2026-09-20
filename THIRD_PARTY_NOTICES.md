# Third-Party Notices

This repository contains original work and modifications by allmkym, and also
contains portions adapted from upstream MIT-licensed code as described below.

## karpathy/llm.c

Upstream project: https://github.com/karpathy/llm.c

Pinned reference used by this project:
`f1e2ace651495b74ae22d45d1723443fd00ecd3a`

The GPT-2 checkpoint conventions, parameter ordering, full-prefix inference
structure, and portions of the scalar kernels in this repository were adapted
from `karpathy/llm.c`. The current project adds C++20 ownership and lifetime
boundaries, validation and error handling, incremental KV-cache inference,
persistent-worker parallel linear execution, tests, benchmarks, and other
engineering changes.

The JYY OS 2026 M6 exercise used an inference-focused subset of `llm.c` as
course material. This repository does not include my original M6 submission or
the JYY course framework.

The upstream `llm.c` license is reproduced below:

MIT License

Copyright (c) 2024 Andrej Karpathy

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Design references not vendored into this repository

The project also references `karpathy/llama2.c` for fixed-capacity
incremental-state design ideas and `ggml` for GPT-2 past-positioning design
context. Their source trees are not included in this repository, and these
references are not intended to imply that their source code is vendored here.

See `IMPLEMENTATION_NOTES.md` for the pinned revisions and source links.

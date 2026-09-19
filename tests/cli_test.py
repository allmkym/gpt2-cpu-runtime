#!/usr/bin/env python3
"""Black-box generation CLI boundary tests with a tiny version-3 checkpoint."""

from __future__ import annotations

import struct
import subprocess
import sys
import tempfile
from pathlib import Path


def parameter_count(max_seq: int, vocab: int, padded: int, layers: int, channels: int) -> int:
    c3 = 3 * channels
    c4 = 4 * channels
    shapes = (
        (padded, channels),
        (max_seq, channels),
        (layers, channels),
        (layers, channels),
        (layers, c3, channels),
        (layers, c3),
        (layers, channels, channels),
        (layers, channels),
        (layers, channels),
        (layers, channels),
        (layers, c4, channels),
        (layers, c4),
        (layers, channels, c4),
        (layers, channels),
        (channels,),
        (channels,),
    )
    total = 0
    for shape in shapes:
        count = 1
        for dimension in shape:
            count *= dimension
        total += count
    return total


def write_zero_checkpoint(path: Path) -> None:
    max_seq, vocab, padded, layers, heads, channels = 4, 5, 8, 2, 2, 4
    header = [0] * 256
    header[:8] = [20240326, 3, max_seq, vocab, layers, heads, channels, padded]
    count = parameter_count(max_seq, vocab, padded, layers, channels)
    path.write_bytes(struct.pack("<256i", *header) + bytes(count * 4))


def run(executable: Path, checkpoint: Path, *arguments: str, expect_success: bool = True):
    result = subprocess.run(
        [str(executable), "--checkpoint", str(checkpoint), *arguments],
        check=False,
        capture_output=True,
        text=True,
    )
    if expect_success and result.returncode != 0:
        raise AssertionError(f"command failed: {result.stderr}")
    if not expect_success and result.returncode == 0:
        raise AssertionError(f"command unexpectedly succeeded: {result.stdout}")
    return result


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("usage: cli_test.py GPT2_GENERATE")
    executable = Path(sys.argv[1])
    with tempfile.TemporaryDirectory(prefix="gpt2_cli_test_") as temporary:
        directory = Path(temporary)
        checkpoint = directory / "tiny_v3.bin"
        write_zero_checkpoint(checkpoint)

        common = ("--tokens", "1,2", "--max-new-tokens", "0")
        full_zero = run(executable, checkpoint, *common, "--mode", "full")
        cached_zero = run(executable, checkpoint, *common, "--mode", "cached", "--capacity", "2")
        assert "output_tokens: 1 2" in full_zero.stdout
        assert "output_tokens: 1 2" in cached_zero.stdout
        assert "step " not in cached_zero.stdout

        generated = ("--tokens", "1,2", "--max-new-tokens", "2", "--greedy")
        full = run(executable, checkpoint, *generated, "--mode", "full")
        default_full = run(executable, checkpoint, *generated)
        cached = run(
            executable, checkpoint, *generated, "--mode", "cached", "--capacity", "3"
        )
        assert "output_tokens: 1 2 0 0" in full.stdout
        assert default_full.stdout == full.stdout
        assert "output_tokens: 1 2 0 0" in cached.stdout
        threaded = run(
            executable,
            checkpoint,
            *generated,
            "--mode",
            "cached",
            "--capacity",
            "3",
            "--threads",
            "2",
        )
        assert threaded.stdout == cached.stdout

        one_generated = ("--tokens", "1,2", "--max-new-tokens", "1", "--greedy")
        cached_one = run(
            executable,
            checkpoint,
            *one_generated,
            "--mode",
            "cached",
            "--capacity",
            "2",
        )
        assert "output_tokens: 1 2 0" in cached_one.stdout

        one_dump = directory / "one_cached.bin"
        one_dump_too_small = run(
            executable,
            checkpoint,
            *one_generated,
            "--mode",
            "cached",
            "--capacity",
            "2",
            "--dump-last-logits",
            str(one_dump),
            expect_success=False,
        )
        assert "session capacity is too small" in one_dump_too_small.stderr
        assert not one_dump.exists()
        run(
            executable,
            checkpoint,
            *one_generated,
            "--mode",
            "cached",
            "--capacity",
            "3",
            "--dump-last-logits",
            str(one_dump),
        )
        assert one_dump.stat().st_size == 5 * 4

        full_dump = directory / "full.bin"
        cached_dump = directory / "cached.bin"
        run(
            executable,
            checkpoint,
            *generated,
            "--mode",
            "full",
            "--dump-last-logits",
            str(full_dump),
        )
        run(
            executable,
            checkpoint,
            *generated,
            "--mode",
            "cached",
            "--capacity",
            "4",
            "--dump-last-logits",
            str(cached_dump),
        )
        assert full_dump.stat().st_size == 8 * 4
        assert cached_dump.stat().st_size == 5 * 4
        assert full_dump.read_bytes()[: 5 * 4] == cached_dump.read_bytes()

        too_small = run(
            executable,
            checkpoint,
            *generated,
            "--mode",
            "cached",
            "--capacity",
            "3",
            "--dump-last-logits",
            str(cached_dump),
            expect_success=False,
        )
        assert "session capacity is too small" in too_small.stderr

    print("generation CLI mode, capacity, dump, and token-count boundary tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

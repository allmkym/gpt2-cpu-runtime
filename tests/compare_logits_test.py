#!/usr/bin/env python3
"""Regression tests for validation/compare_logits.py failure boundaries."""

from __future__ import annotations

import struct
import subprocess
import sys
import tempfile
from pathlib import Path


def invoke(script: Path, candidate: Path, reference: Path, *options: str):
    return subprocess.run(
        [
            sys.executable,
            str(script),
            str(candidate),
            str(reference),
            "--expected-count",
            "3",
            *options,
        ],
        check=False,
        capture_output=True,
        text=True,
    )


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("usage: compare_logits_test.py COMPARE_SCRIPT")
    script = Path(sys.argv[1])
    with tempfile.TemporaryDirectory(prefix="gpt2_compare_test_") as temporary:
        directory = Path(temporary)
        reference = directory / "reference.bin"
        candidate = directory / "candidate.bin"
        reference.write_bytes(struct.pack("<3f", 1.0, 3.0, -2.0))
        candidate.write_bytes(reference.read_bytes())

        exact = invoke(script, candidate, reference, "--require-bit-identical")
        assert exact.returncode == 0, exact.stderr
        assert "byte_identical=yes" in exact.stdout
        assert "candidate_argmax=1" in exact.stdout

        candidate.write_bytes(struct.pack("<3f", 1.00001, 3.0, -2.0))
        tolerant = invoke(script, candidate, reference, "--absolute-tolerance", "1e-4")
        assert tolerant.returncode == 0, tolerant.stdout + tolerant.stderr
        assert "byte_identical=no" in tolerant.stdout
        exact_failure = invoke(
            script,
            candidate,
            reference,
            "--absolute-tolerance",
            "1e-4",
            "--require-bit-identical",
        )
        assert exact_failure.returncode == 1
        assert "raw bytes differ" in exact_failure.stdout

        wrong_count = subprocess.run(
            [
                sys.executable,
                str(script),
                str(candidate),
                str(reference),
                "--expected-count",
                "4",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        assert wrong_count.returncode != 0
        assert "element count mismatch" in wrong_count.stderr

        empty = directory / "empty.bin"
        empty.write_bytes(b"")
        empty_failure = invoke(script, empty, reference)
        assert empty_failure.returncode != 0
        assert "is empty" in empty_failure.stderr

        invalid_tolerance = invoke(
            script, candidate, reference, "--absolute-tolerance", "nan"
        )
        assert invalid_tolerance.returncode != 0
        assert "finite and nonnegative" in invalid_tolerance.stderr

    print("compare_logits validation boundary tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

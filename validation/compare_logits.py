#!/usr/bin/env python3
"""Compare two nonempty raw little-endian float32 logits files."""

from __future__ import annotations

import argparse
import math
import struct
from pathlib import Path


def read_floats(path: Path) -> tuple[bytes, tuple[float, ...]]:
    payload = path.read_bytes()
    if not payload:
        raise ValueError(f"{path} is empty")
    if len(payload) % 4:
        raise ValueError(f"{path} is not a float32 array")
    return payload, struct.unpack(f"<{len(payload) // 4}f", payload)


def valid_tolerance(value: float, name: str) -> float:
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return value


def argmax(values: tuple[float, ...]) -> int:
    return max(range(len(values)), key=values.__getitem__)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("--expected-count", type=int, required=True)
    parser.add_argument("--absolute-tolerance", type=float, default=1.0e-4)
    parser.add_argument("--relative-tolerance", type=float, default=0.0)
    parser.add_argument("--require-bit-identical", action="store_true")
    args = parser.parse_args()

    if args.expected_count <= 0:
        raise ValueError("--expected-count must be positive")
    absolute_tolerance = valid_tolerance(
        args.absolute_tolerance, "--absolute-tolerance"
    )
    relative_tolerance = valid_tolerance(
        args.relative_tolerance, "--relative-tolerance"
    )

    candidate_bytes, candidate = read_floats(args.candidate)
    reference_bytes, reference = read_floats(args.reference)
    if len(candidate) != len(reference):
        raise ValueError(f"length mismatch: {len(candidate)} != {len(reference)}")
    if len(candidate) != args.expected_count:
        raise ValueError(
            f"element count mismatch: {len(candidate)} != {args.expected_count}"
        )

    max_absolute = 0.0
    max_absolute_index = 0
    max_relative = 0.0
    max_relative_index = 0
    violations = 0
    first_violation = None
    for index, (actual, expected) in enumerate(zip(candidate, reference, strict=True)):
        if not math.isfinite(actual) or not math.isfinite(expected):
            raise ValueError(f"non-finite value at index {index}")
        absolute = abs(actual - expected)
        relative = absolute / max(abs(expected), 1.0e-6)
        if absolute > max_absolute:
            max_absolute = absolute
            max_absolute_index = index
        if relative > max_relative:
            max_relative = relative
            max_relative_index = index
        threshold = absolute_tolerance + relative_tolerance * abs(expected)
        if absolute > threshold:
            violations += 1
            if first_violation is None:
                first_violation = index

    byte_identical = candidate_bytes == reference_bytes
    candidate_argmax = argmax(candidate)
    reference_argmax = argmax(reference)
    print(f"compared_values={len(candidate)}")
    print(f"byte_identical={'yes' if byte_identical else 'no'}")
    print(f"max_absolute_error={max_absolute:.9g} index={max_absolute_index}")
    print(f"max_relative_error={max_relative:.9g} index={max_relative_index}")
    print(f"candidate_argmax={candidate_argmax}")
    print(f"reference_argmax={reference_argmax}")
    print(f"absolute_tolerance={absolute_tolerance:.9g}")
    print(f"relative_tolerance={relative_tolerance:.9g}")
    print(f"tolerance_violations={violations}")
    if first_violation is not None:
        print(f"first_tolerance_violation={first_violation}")

    if args.require_bit_identical and not byte_identical:
        print("result=FAIL (raw bytes differ)")
        return 1
    if violations != 0:
        print("result=FAIL (tolerance exceeded)")
        return 1
    if candidate_argmax != reference_argmax:
        print("result=FAIL (argmax differs)")
        return 1
    print("result=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

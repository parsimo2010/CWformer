#!/usr/bin/env python3
"""
CER.py — Character Error Rate between a target string and a hypothesis.

Usage:
    python CER.py --target "CQ CQ DE W1AW" --string "CQ CJ DE WHAW"

Both sides are stripped and upper-cased before comparison. CER is the
Levenshtein edit distance (insertions + deletions + substitutions)
divided by the length of the target.
"""

from __future__ import annotations

import argparse


def levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        return levenshtein(b, a)
    if len(b) == 0:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        curr = [i + 1]
        for j, cb in enumerate(b):
            cost = 0 if ca == cb else 1
            curr.append(min(curr[j] + 1, prev[j + 1] + 1, prev[j] + cost))
        prev = curr
    return prev[-1]


def compute_cer(hypothesis: str, reference: str) -> tuple[float, int, int]:
    h = hypothesis.strip().upper()
    r = reference.strip().upper()
    if not r:
        return (0.0 if not h else 1.0, 0, len(r))
    edits = levenshtein(h, r)
    return edits / len(r), edits, len(r)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute CER between a target and a decoded string.")
    parser.add_argument("--target", required=True,
                        help="Ground-truth text.")
    parser.add_argument("--string", required=True, dest="hypothesis",
                        help="Decoded / hypothesis text.")
    args = parser.parse_args()

    cer, edits, ref_len = compute_cer(args.hypothesis, args.target)
    print(f"Target ({ref_len} chars): {args.target.strip().upper()}")
    print(f"String ({len(args.hypothesis.strip())} chars): {args.hypothesis.strip().upper()}")
    print(f"Edit distance:  {edits}")
    print(f"CER:            {cer:.4f}  ({cer:.2%})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

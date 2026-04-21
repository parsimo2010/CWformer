"""
metrics.py — Shared Levenshtein / CER helpers.

All CER computation in this repo uses these functions so training logs,
benchmarks, and demo tools report comparable numbers.

Conventions:
  - Both sides are ``.strip().upper()``-normalised before comparison.
  - CER = edit distance divided by reference length.
  - An empty reference yields CER = 0 if the hypothesis is also empty,
    else 1.0.
"""

from __future__ import annotations

from typing import List


def levenshtein(a: str, b: str) -> int:
    """Classical edit distance (insertions + deletions + substitutions)."""
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


def compute_cer(hypothesis: str, reference: str) -> float:
    """Character Error Rate. Both sides stripped and upper-cased."""
    h = hypothesis.strip().upper()
    r = reference.strip().upper()
    if not r:
        return 0.0 if not h else 1.0
    return levenshtein(h, r) / len(r)


def per_position_errors(hypothesis: str, reference: str) -> List[bool]:
    """Per-reference-character correctness flags via Levenshtein alignment.

    Returns a list of booleans, one per reference character. ``True`` means
    that character was matched exactly in the optimal alignment; ``False``
    means it was substituted or deleted. Uses the standard DP matrix
    backtrace to find the alignment.
    """
    h = hypothesis.strip().upper()
    r = reference.strip().upper()
    if not r:
        return []

    n, m = len(r), len(h)

    # Build DP matrix
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if r[i - 1] == h[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,          # deletion (ref char skipped)
                dp[i][j - 1] + 1,          # insertion (extra hyp char)
                dp[i - 1][j - 1] + cost,   # match/substitution
            )

    # Backtrace to get per-ref-char correct/error
    correct = [False] * n
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            cost = 0 if r[i - 1] == h[j - 1] else 1
            if dp[i][j] == dp[i - 1][j - 1] + cost:
                if cost == 0:
                    correct[i - 1] = True
                i -= 1
                j -= 1
                continue
        if j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            j -= 1  # insertion — skip hyp char
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            i -= 1  # deletion — ref char missed, correct[i] stays False
        else:
            # Defensive: shouldn't be reachable given the DP recurrence.
            if i > 0:
                i -= 1
            if j > 0:
                j -= 1

    return correct

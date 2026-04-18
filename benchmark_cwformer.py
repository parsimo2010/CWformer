#!/usr/bin/env python3
"""
benchmark_cwformer.py — Controlled CW-Former accuracy profiling.

Phase 1: Clean baseline grid (SNR x WPM x key type, no augmentations)
Phase 2: Augmentation impact (one at a time, controlled parameters)

All results logged to CSV for future analysis.  Uses greedy CTC decoding.

Usage:
    python benchmark_cwformer.py
    python benchmark_cwformer.py --device cuda --samples 10
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

from config import MorseConfig
from morse_generator import generate_sample
from neural_decoder.inference_cwformer import CWFormerStreamingDecoder


# ---------------------------------------------------------------------------
# CER computation (matches train_cwformer.py)
# ---------------------------------------------------------------------------

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


def compute_cer(hypothesis: str, reference: str) -> float:
    """Character Error Rate.  Both sides stripped and upper-cased."""
    h = hypothesis.strip().upper()
    r = reference.strip().upper()
    if not r:
        return 0.0 if not h else 1.0
    return levenshtein(h, r) / len(r)


def per_position_errors(hypothesis: str, reference: str) -> List[bool]:
    """Compute per-reference-character error flags via Levenshtein alignment.

    Returns a list of booleans, one per reference character, where True
    means that character was correctly matched in the alignment and False
    means it was substituted or deleted.

    Uses the standard DP matrix backtrace to find the optimal alignment.
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
                dp[i - 1][j] + 1,      # deletion (ref char skipped)
                dp[i][j - 1] + 1,       # insertion (extra hyp char)
                dp[i - 1][j - 1] + cost  # match/substitution
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
            # Shouldn't happen, but defensive
            if i > 0:
                i -= 1
            if j > 0:
                j -= 1

    return correct


# ---------------------------------------------------------------------------
# Config builders
# ---------------------------------------------------------------------------

KEY_WEIGHTS = {
    "straight": (1.0, 0.0, 0.0, 0.0),
    "bug":      (0.0, 1.0, 0.0, 0.0),
    "paddle":   (0.0, 0.0, 1.0, 0.0),
    "cootie":   (0.0, 0.0, 0.0, 1.0),
}


def _base_config() -> MorseConfig:
    """MorseConfig with all augmentations OFF, full-scenario timing."""
    mc = MorseConfig()
    # Full-scenario timing (operator style variance)
    mc.dah_dit_ratio_min = 1.3
    mc.dah_dit_ratio_max = 4.0
    mc.ics_factor_min = 0.5
    mc.ics_factor_max = 2.0
    mc.iws_factor_min = 0.5
    mc.iws_factor_max = 2.5
    mc.timing_jitter = 0.0
    mc.timing_jitter_max = 0.25
    mc.tone_drift = 5.0
    mc.rise_time_ms_min = 3.0
    mc.rise_time_ms_max = 8.0
    # Most augmentations OFF — except bandpass, which is baseline because
    # a real receiver always applies a CW filter. Pinned 250-600 Hz, order
    # 4-8, which matches a typical amateur CW receiver.
    mc.agc_probability = 0.0
    mc.qsb_probability = 0.0
    mc.qrm_probability = 0.0
    mc.qrn_probability = 0.0
    mc.bandpass_probability = 1.0
    mc.bandpass_bw_min = 250.0
    mc.bandpass_bw_max = 600.0
    mc.bandpass_order_min = 4
    mc.bandpass_order_max = 8
    mc.hf_noise_probability = 0.0
    mc.farnsworth_probability = 0.0
    mc.multi_op_probability = 0.0
    mc.speed_drift_max = 0.0
    # Sample length — long audio to match real-world deployment captures.
    # At 35 WPM (highest tested): 30 s minimum => 88+ chars, round to 90.
    # At 15 WPM (lowest tested): max 150 chars => 120 s, bounded.
    mc.min_chars = 90
    mc.max_chars = 150
    return mc


def make_config(
    snr_db: float, wpm: float, key_type: str,
    aug_overrides: dict | None = None,
) -> MorseConfig:
    """Build a config with pinned SNR/WPM/key_type and optional augmentation."""
    mc = _base_config()
    mc.min_snr_db = snr_db
    mc.max_snr_db = snr_db
    mc.min_wpm = wpm
    mc.max_wpm = wpm
    mc.key_type_weights = KEY_WEIGHTS[key_type]
    if aug_overrides:
        for k, v in aug_overrides.items():
            setattr(mc, k, v)
    return mc


# ---------------------------------------------------------------------------
# Augmentation specs for Phase 2
# ---------------------------------------------------------------------------

# Each entry: (label, {config overrides})
# Probability=1.0 forces the augmentation on; min=max pins the value.
AUGMENTATIONS = [
    ("AGC 12 dB", {
        "agc_probability": 1.0,
        "agc_depth_db_min": 12.0, "agc_depth_db_max": 12.0,
    }),
    ("AGC 20 dB", {
        "agc_probability": 1.0,
        "agc_depth_db_min": 20.0, "agc_depth_db_max": 20.0,
    }),
    ("QSB 8 dB", {
        "qsb_probability": 1.0,
        "qsb_depth_db_min": 8.0, "qsb_depth_db_max": 8.0,
    }),
    ("QSB 16 dB", {
        "qsb_probability": 1.0,
        "qsb_depth_db_min": 16.0, "qsb_depth_db_max": 16.0,
    }),
    ("QRM x1", {
        "qrm_probability": 1.0,
        "qrm_count_min": 1, "qrm_count_max": 1,
        "qrm_amplitude_min": 0.3, "qrm_amplitude_max": 0.5,
    }),
    ("QRM x2", {
        "qrm_probability": 1.0,
        "qrm_count_min": 2, "qrm_count_max": 2,
        "qrm_amplitude_min": 0.3, "qrm_amplitude_max": 0.5,
    }),
    ("QRN rate=2", {
        "qrn_probability": 1.0,
        "qrn_rate_min": 2.0, "qrn_rate_max": 2.0,
    }),
    ("QRN rate=5", {
        "qrn_probability": 1.0,
        "qrn_rate_min": 5.0, "qrn_rate_max": 5.0,
    }),
    ("Drift 8%", {
        "speed_drift_max": 0.08,
    }),
    ("Drift 15%", {
        "speed_drift_max": 0.15,
    }),
    ("Farnsworth 1.5x", {
        "farnsworth_probability": 1.0,
        "farnsworth_char_speed_min": 1.5, "farnsworth_char_speed_max": 1.5,
    }),
    ("Farnsworth 2.0x", {
        "farnsworth_probability": 1.0,
        "farnsworth_char_speed_min": 2.0, "farnsworth_char_speed_max": 2.0,
    }),
]


# ---------------------------------------------------------------------------
# CSV logging
# ---------------------------------------------------------------------------

CSV_FIELDS = [
    "phase", "condition", "snr_db", "wpm", "key_type",
    "augmentation", "sample_idx",
    "ref_text", "hyp_text", "cer", "ref_len", "audio_sec",
    "dah_dit_ratio", "ics_factor", "iws_factor", "timing_jitter",
    "tone_freq_hz", "agc_depth_db", "qsb_depth_db",
    "qrm", "qrm_count", "qrn", "bandpass", "bandpass_bw",
    "farnsworth_stretch", "speed_drift_max",
]


def _meta_row(phase, condition, aug_label, sample_idx,
              mc, meta, ref_text, hyp_text, cer):
    return {
        "phase": phase,
        "condition": condition,
        "snr_db": meta["snr_db"],
        "wpm": meta["wpm"],
        "key_type": meta["key_type"],
        "augmentation": aug_label,
        "sample_idx": sample_idx,
        "ref_text": ref_text,
        "hyp_text": hyp_text,
        "cer": f"{cer:.4f}",
        "ref_len": len(ref_text.strip()),
        "audio_sec": f"{meta['duration_sec']:.1f}",
        "dah_dit_ratio": f"{meta['dah_dit_ratio']:.2f}",
        "ics_factor": f"{meta['ics_factor']:.2f}",
        "iws_factor": f"{meta['iws_factor']:.2f}",
        "timing_jitter": f"{meta['timing_jitter']:.3f}",
        "tone_freq_hz": f"{meta['base_frequency_hz']:.0f}",
        "agc_depth_db": f"{meta['agc_depth_db']:.1f}",
        "qsb_depth_db": f"{meta['qsb_depth_db']:.1f}",
        "qrm": meta["qrm"],
        "qrm_count": meta["qrm_count"],
        "qrn": meta["qrn"],
        "bandpass": meta["bandpass"],
        "bandpass_bw": f"{meta['bandpass_bw']:.0f}",
        "farnsworth_stretch": f"{meta['farnsworth_stretch']:.2f}",
        "speed_drift_max": mc.speed_drift_max,
    }


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------

def eval_cell(
    decoder: CWFormerStreamingDecoder,
    mc: MorseConfig,
    n_samples: int,
    seed: int,
    csv_writer,
    phase: str,
    condition: str,
    aug_label: str,
) -> List[float]:
    """Generate samples, decode, compute CER, log to CSV.  Returns CER list."""
    rng = np.random.default_rng(seed)
    cers = []
    for i in range(n_samples):
        try:
            audio, text, meta = generate_sample(mc, rng=rng)
        except Exception as e:
            print(f"    WARN: gen failed: {e}", file=sys.stderr)
            cers.append(1.0)
            continue
        hyp = decoder.decode_audio(audio)
        cer = compute_cer(hyp, text)
        cers.append(cer)
        if csv_writer:
            csv_writer.writerow(
                _meta_row(phase, condition, aug_label, i, mc, meta, text, hyp, cer)
            )
    return cers




# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Benchmark CW-Former (controlled)")
    parser.add_argument("--checkpoint", default="checkpoints_cwformer_full/best_model.pt")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--samples", type=int, default=10,
                        help="Samples per parameter combination")
    parser.add_argument("--csv", default="benchmark_results.csv",
                        help="Output CSV path for per-sample results")
    parser.add_argument("--chunk-ms", type=int, default=500, dest="chunk_ms",
                        help="Streaming chunk size in milliseconds")
    args = parser.parse_args()

    csv_fh = open(args.csv, "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(csv_fh, fieldnames=CSV_FIELDS)
    writer.writeheader()

    dec = CWFormerStreamingDecoder(
        checkpoint=args.checkpoint,
        chunk_ms=args.chunk_ms,
        device=args.device,
    )
    n = args.samples

    # ==================================================================
    # Phase 1: Clean baseline (no augmentations)
    # ==================================================================
    snr_levels = [30, 20, 10, 5, 0, -5]
    wpm_levels = [15, 20, 25, 30, 35]
    key_types = ["straight", "bug", "paddle", "cootie"]

    total_p1 = len(snr_levels) * len(wpm_levels) * len(key_types)
    print("=" * 72)
    print(f"Phase 1: Clean baseline ({total_p1} cells x {n} samples, greedy)")
    print("  No augmentations — pure CW signal + AWGN")
    print("=" * 72)

    p1: Dict[Tuple, List[float]] = {}
    idx = 0
    t0 = time.time()

    for snr in snr_levels:
        for wpm in wpm_levels:
            for kt in key_types:
                idx += 1
                seed = 20000 + (snr + 10) * 1000 + wpm * 10 + key_types.index(kt)
                mc = make_config(snr, wpm, kt)
                cond = f"SNR={snr} WPM={wpm} {kt}"

                cers = eval_cell(dec, mc, n, seed, writer,
                                 phase="baseline", condition=cond, aug_label="none")
                p1[(snr, wpm, kt)] = cers
                mean = np.mean(cers)
                elapsed = time.time() - t0
                eta = elapsed / idx * (total_p1 - idx)
                print(f"  [{idx:3d}/{total_p1}] {cond:<30s} "
                      f"CER={mean:5.1%}  (ETA {eta/60:.0f}m)", flush=True)

    csv_fh.flush()

    # --- Phase 1 summary tables ---
    print("\n" + "=" * 72)
    print("Phase 1 Results: Clean baseline")
    print("=" * 72)

    # Table: SNR x WPM (averaged over key types)
    print(f"\n{'SNR':>5s}", end="")
    for wpm in wpm_levels:
        print(f" {wpm:>5d}", end="")
    print("   Avg")
    print("-" * (5 + 6 * len(wpm_levels) + 6))

    for snr in snr_levels:
        print(f"{snr:4d} ", end="")
        row = []
        for wpm in wpm_levels:
            cell = []
            for kt in key_types:
                cell.extend(p1[(snr, wpm, kt)])
            row.extend(cell)
            print(f"{np.mean(cell):5.1%} ", end="")
        print(f" {np.mean(row):5.1%}")

    print(f"{'Avg':>5s}", end="")
    for wpm in wpm_levels:
        col = []
        for snr in snr_levels:
            for kt in key_types:
                col.extend(p1[(snr, wpm, kt)])
        print(f" {np.mean(col):5.1%}", end="")
    all_p1 = [c for v in p1.values() for c in v]
    print(f"  {np.mean(all_p1):5.1%}")

    # Table: Key type x SNR (averaged over WPM)
    print(f"\n{'Key':>10s}", end="")
    for snr in snr_levels:
        print(f" {snr:>5d}", end="")
    print("   Avg")
    print("-" * (10 + 6 * len(snr_levels) + 6))

    for kt in key_types:
        print(f"{kt:>10s}", end="")
        row = []
        for snr in snr_levels:
            cell = []
            for wpm in wpm_levels:
                cell.extend(p1[(snr, wpm, kt)])
            row.extend(cell)
            print(f" {np.mean(cell):5.1%}", end="")
        print(f"  {np.mean(row):5.1%}")

    # Full breakdown
    print(f"\n{'SNR':>4s} {'WPM':>4s}", end="")
    for kt in key_types:
        print(f" {kt:>8s}", end="")
    print("    Avg")
    print("-" * (8 + 9 * len(key_types) + 7))

    for snr in snr_levels:
        for wpm in wpm_levels:
            print(f"{snr:4d} {wpm:4d}", end="")
            row = []
            for kt in key_types:
                cell = p1[(snr, wpm, kt)]
                row.extend(cell)
                print(f" {np.mean(cell):7.1%}", end="")
            print(f"  {np.mean(row):5.1%}")
        if snr != snr_levels[-1]:
            print()

    p1_time = time.time() - t0
    print(f"\nPhase 1 overall CER: {np.mean(all_p1):.1%}  ({p1_time/60:.1f} min)")

    # ==================================================================
    # Phase 2: Augmentation impact
    #   Fixed: SNR=20, WPM=25, all key types
    #   Each augmentation one at a time
    # ==================================================================
    aug_snr = 20
    aug_wpm = 25

    total_p2 = len(AUGMENTATIONS) * len(key_types)
    print("\n" + "=" * 72)
    print(f"Phase 2: Augmentation impact ({total_p2} cells x {n} samples)")
    print(f"  Fixed: SNR={aug_snr} dB, WPM={aug_wpm}")
    print(f"  Baseline CER (from Phase 1) shown for comparison")
    print("=" * 72)

    # Baseline from Phase 1 at (aug_snr, aug_wpm)
    baseline_by_kt = {}
    for kt in key_types:
        baseline_by_kt[kt] = np.mean(p1[(aug_snr, aug_wpm, kt)])

    baseline_avg = np.mean([baseline_by_kt[kt] for kt in key_types])
    print(f"\n  Baseline (no aug): {baseline_avg:5.1%}  "
          f"[{', '.join(f'{kt}={baseline_by_kt[kt]:.1%}' for kt in key_types)}]\n")

    p2: Dict[Tuple[str, str], List[float]] = {}
    idx = 0
    t2 = time.time()

    for aug_label, aug_params in AUGMENTATIONS:
        for kt in key_types:
            idx += 1
            # Same seed family as Phase 1 so underlying text/timing match
            seed = 2000 + aug_snr * 1000 + aug_wpm * 10 + key_types.index(kt)
            mc = make_config(aug_snr, aug_wpm, kt, aug_overrides=aug_params)
            cond = f"{aug_label} {kt}"

            cers = eval_cell(dec, mc, n, seed, writer,
                             phase="augmentation", condition=cond, aug_label=aug_label)
            p2[(aug_label, kt)] = cers
            mean = np.mean(cers)
            delta = mean - baseline_by_kt[kt]
            eta = (time.time() - t2) / idx * (total_p2 - idx)
            print(f"  [{idx:3d}/{total_p2}] {cond:<28s} "
                  f"CER={mean:5.1%}  ({delta:+5.1%} vs base)  "
                  f"(ETA {eta/60:.0f}m)", flush=True)

    csv_fh.flush()

    # --- Phase 2 summary ---
    print("\n" + "=" * 72)
    print("Phase 2 Results: Augmentation impact")
    print(f"  Base condition: SNR={aug_snr}, WPM={aug_wpm}, no augmentations")
    print("=" * 72)

    print(f"\n{'Augmentation':<20s}", end="")
    for kt in key_types:
        print(f" {kt:>8s}", end="")
    print("      Avg   Delta")
    print("-" * (20 + 9 * len(key_types) + 14))

    # Baseline row
    print(f"{'(baseline)':<20s}", end="")
    for kt in key_types:
        print(f" {baseline_by_kt[kt]:7.1%}", end="")
    print(f"    {baseline_avg:5.1%}    ---")

    for aug_label, _ in AUGMENTATIONS:
        print(f"{aug_label:<20s}", end="")
        aug_cers_all = []
        for kt in key_types:
            cell = p2[(aug_label, kt)]
            aug_cers_all.extend(cell)
            print(f" {np.mean(cell):7.1%}", end="")
        aug_avg = np.mean(aug_cers_all)
        delta = aug_avg - baseline_avg
        print(f"    {aug_avg:5.1%}  {delta:+5.1%}")

    p2_time = time.time() - t2
    csv_fh.flush()

    # ==================================================================
    # Phase 3: Per-character-position error rate (context ramp-up)
    #   The causal model processes audio left-to-right. The first few
    #   characters are decoded with minimal past context — no prior
    #   examples of the operator's fist. Later characters benefit from
    #   a full context window. This phase measures where accuracy
    #   stabilizes, answering: "how many characters does the model
    #   need before it locks onto the operator's style?"
    #
    #   For each sample, Levenshtein alignment classifies each reference
    #   character as correct or error. Aggregating by position across
    #   many samples gives an error-rate-by-position curve.
    # ==================================================================
    pos_snr_levels = [10, 20, 30]
    pos_wpm_levels = [15, 20, 25, 30]
    pos_n = max(n, 20)  # need enough samples for stable position stats
    max_pos = 80  # track up to 80 characters deep

    total_p3 = len(pos_snr_levels) * len(pos_wpm_levels)
    print("\n" + "=" * 72)
    print(f"Phase 3: Per-character-position error rate (context ramp-up)")
    print(f"  {total_p3} conditions x {pos_n} samples, paddle key")
    print(f"  Measures error rate at each character position (1st, 2nd, ...)")
    print("=" * 72)

    # pos_errors[pos] = list of bools (True=correct, False=error) across all samples
    pos_errors_all: Dict[int, List[bool]] = {}
    # Also track by WPM: pos_errors_by_wpm[wpm][pos] = list of bools
    pos_errors_by_wpm: Dict[int, Dict[int, List[bool]]] = {}

    idx = 0
    t3 = time.time()

    for snr in pos_snr_levels:
        for wpm in pos_wpm_levels:
            idx += 1
            mc = make_config(snr, wpm, "paddle")
            rng = np.random.default_rng(40000 + snr * 100 + wpm)

            if wpm not in pos_errors_by_wpm:
                pos_errors_by_wpm[wpm] = {}

            for si in range(pos_n):
                try:
                    audio, text, meta = generate_sample(mc, rng=rng)
                except Exception:
                    continue

                hyp = dec.decode_audio(audio)
                flags = per_position_errors(hyp, text)

                for pos_idx, is_correct in enumerate(flags):
                    if pos_idx >= max_pos:
                        break
                    pos_errors_all.setdefault(pos_idx, []).append(is_correct)
                    pos_errors_by_wpm[wpm].setdefault(pos_idx, []).append(is_correct)

            elapsed = time.time() - t3
            eta = elapsed / idx * (total_p3 - idx) if idx > 0 else 0
            print(f"  [{idx:2d}/{total_p3}] SNR={snr:3d} WPM={wpm:2d}  "
                  f"(ETA {eta/60:.0f}m)", flush=True)

    # --- Phase 3 summary ---
    print("\n" + "=" * 72)
    print("Phase 3 Results: Error rate by character position")
    print("  (lower = better; shows context ramp-up curve)")
    print("=" * 72)

    # Overall curve
    print(f"\n  Overall (all SNR/WPM averaged):")
    print(f"  {'Pos':>4s}  {'ErrRate':>7s}  {'N':>5s}  {'Bar'}")
    print(f"  {'-'*4}  {'-'*7}  {'-'*5}  {'-'*40}")

    for pos_idx in range(max_pos):
        flags = pos_errors_all.get(pos_idx, [])
        if len(flags) < 5:
            break
        err_rate = 1.0 - (sum(flags) / len(flags))
        bar_len = int(err_rate * 40)
        bar = "#" * bar_len
        print(f"  {pos_idx + 1:4d}  {err_rate:6.1%}  {len(flags):5d}  {bar}")

    # By-WPM curve (compact: show first 20 positions)
    show_positions = min(20, max_pos)
    print(f"\n  Error rate by WPM (first {show_positions} character positions):")
    print(f"  {'WPM':>4s}", end="")
    for p in range(show_positions):
        print(f" {p+1:5d}", end="")
    print()
    print(f"  {'':>4s}", end="")
    for p in range(show_positions):
        print(f" {'-----'}", end="")
    print()

    for wpm in sorted(pos_errors_by_wpm.keys()):
        wpm_data = pos_errors_by_wpm[wpm]
        print(f"  {wpm:4d}", end="")
        for p in range(show_positions):
            flags = wpm_data.get(p, [])
            if len(flags) >= 3:
                err = 1.0 - (sum(flags) / len(flags))
                print(f" {err:4.0%}", end=" ")
            else:
                print(f"   -- ", end="")
        print()

    # Find stabilization point: first position where error rate drops
    # below 1.2x the average error rate of positions 20-60
    late_errors = []
    for p in range(20, 60):
        flags = pos_errors_all.get(p, [])
        if len(flags) >= 5:
            late_errors.append(1.0 - sum(flags) / len(flags))
    if late_errors:
        baseline_err = np.mean(late_errors)
        threshold = baseline_err * 1.2
        stable_pos = None
        for p in range(max_pos):
            flags = pos_errors_all.get(p, [])
            if len(flags) < 5:
                break
            err = 1.0 - sum(flags) / len(flags)
            if err <= threshold:
                stable_pos = p + 1
                break
        print(f"\n  Steady-state error rate (pos 21-60): {baseline_err:.1%}")
        if stable_pos is not None:
            print(f"  Error drops to within 1.2x steady-state at: character {stable_pos}")
        else:
            print(f"  Error rate never stabilizes to within 1.2x of steady-state")

    p3_time = time.time() - t3
    total_time = time.time() - t0
    print(f"\nPhase 3 time: {p3_time/60:.1f} min")
    print(f"Total time:   {total_time/60:.1f} min")
    print(f"CSV saved to: {args.csv}")


if __name__ == "__main__":
    main()

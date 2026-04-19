#!/usr/bin/env python3
"""
benchmark_mrp40.py — Benchmark MRP40 on synthetic Morse audio.

Pipeline per sample:
  1. Generate synthetic CW audio with morse_generator (same tight-timing
     distribution as benchmark_cwformer.py Phase 1, but shorter samples so
     the run fits in roughly an hour of wall clock).
  2. Play the audio into a virtual audio cable (VB-Cable) that MRP40 is
     using as its input soundcard.
  3. Wait for playback + a flush margin so MRP40 finishes decoding the tail.
  4. Read the new bytes appended to MRP40's log file since the sample
     started. That's the hypothesis.
  5. Compute CER vs ground truth. Log per-sample row to CSV.

One-time Windows setup (manual — cannot be automated by this script):
  a. Install VB-CABLE from https://vb-audio.com/Cable/ (free). This creates
     "CABLE Input" (what we play TO) and "CABLE Output" (what apps listen
     FROM).
  b. In MRP40:   Settings -> Audio In -> "CABLE Output (VB-Audio Virtual
     Cable)".   Enable Settings -> Text log -> Log received text; note
     the log path (pass it as --mrp40-log).
  c. Start MRP40 and let it run for the duration of the benchmark. Leave
     it in its default auto-WPM / auto-threshold mode.

Python deps:  pip install sounddevice numpy

Usage:
  # Discover audio device names first:
  python benchmark_mrp40.py --list-devices

  # Dry-run prints the plan + expected audio duration without playing:
  python benchmark_mrp40.py --mrp40-log "C:/.../mrp40.log" \\
      --output-device "CABLE Input" --dry-run

  # Real run:
  python benchmark_mrp40.py --mrp40-log "C:/.../mrp40.log" \\
      --output-device "CABLE Input" --csv mrp40_results.csv
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from config import MorseConfig
from morse_generator import generate_sample


# ---------------------------------------------------------------------------
# CER (duplicated from benchmark_cwformer.py to keep this module standalone)
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
    h = hypothesis.strip().upper()
    r = reference.strip().upper()
    if not r:
        return 0.0 if not h else 1.0
    return levenshtein(h, r) / len(r)


# ---------------------------------------------------------------------------
# MRP40 log cleaning
# ---------------------------------------------------------------------------

def clean_mrp40_output(raw: str) -> str:
    """Best-effort normalisation of MRP40 log text for CER comparison.

    MRP40's log appends decoded text verbatim but may include prosign
    tokens (<AR>, <BT>, ...) and occasional non-alphanumeric punctuation
    that our ground-truth text does not contain. Strip those, uppercase,
    and collapse whitespace.
    """
    # Drop prosign tokens like <AR>, <SK>, <BT>
    text = re.sub(r"<[^>]*>", " ", raw)
    out = []
    for ch in text.upper():
        if ch.isalnum() or ch.isspace():
            out.append(ch)
        else:
            out.append(" ")
    return " ".join("".join(out).split())


# ---------------------------------------------------------------------------
# Config builders — mirror benchmark_cwformer.py tight-timing baseline
# ---------------------------------------------------------------------------

KEY_WEIGHTS = {
    "straight": (1.0, 0.0, 0.0, 0.0),
    "bug":      (0.0, 1.0, 0.0, 0.0),
    "paddle":   (0.0, 0.0, 1.0, 0.0),
    "cootie":   (0.0, 0.0, 0.0, 1.0),
}


def _base_config(min_chars: int, max_chars: int) -> MorseConfig:
    mc = MorseConfig()
    # Tight operator timing — match benchmark_cwformer.py Phase 1.
    mc.dah_dit_ratio_min = 2.0
    mc.dah_dit_ratio_max = 4.0
    mc.ics_factor_min = 0.75
    mc.ics_factor_max = 1.5
    mc.iws_factor_min = 0.75
    mc.iws_factor_max = 1.75
    mc.timing_jitter = 0.0
    mc.timing_jitter_max = 0.25
    mc.tone_drift = 5.0
    mc.rise_time_ms_min = 3.0
    mc.rise_time_ms_max = 8.0
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
    # Shorter samples than the neural benchmark (90–150) so the 1x-real-time
    # MRP40 pipeline fits in a reasonable wall clock.
    mc.min_chars = min_chars
    mc.max_chars = max_chars
    return mc


def make_config(snr_db, wpm, key_type, min_chars, max_chars) -> MorseConfig:
    mc = _base_config(min_chars, max_chars)
    mc.min_snr_db = snr_db
    mc.max_snr_db = snr_db
    mc.min_wpm = wpm
    mc.max_wpm = wpm
    mc.key_type_weights = KEY_WEIGHTS[key_type]
    return mc


# ---------------------------------------------------------------------------
# Audio device helpers
# ---------------------------------------------------------------------------

def resolve_output_device(name_or_index: str, sd):
    """Resolve --output-device to a PortAudio device index."""
    try:
        return int(name_or_index)
    except ValueError:
        pass
    devs = sd.query_devices()
    matches = [
        i for i, d in enumerate(devs)
        if name_or_index.lower() in d["name"].lower()
        and d["max_output_channels"] > 0
    ]
    if not matches:
        raise ValueError(
            f"No output device matches '{name_or_index}'. "
            f"Run with --list-devices to see available devices."
        )
    if len(matches) > 1:
        names = [f"  [{i}] {devs[i]['name']}" for i in matches]
        raise ValueError(
            f"Multiple output devices match '{name_or_index}':\n"
            + "\n".join(names)
            + "\nBe more specific or pass the index."
        )
    return matches[0]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark MRP40 on synthetic CW audio via VB-Cable."
    )
    parser.add_argument("--list-devices", action="store_true",
                        help="List audio output devices and exit.")
    parser.add_argument("--mrp40-log",
                        help="Path to MRP40's text log file (enable logging "
                             "in MRP40 and note the path).")
    parser.add_argument("--output-device",
                        help="Audio output device index or name substring "
                             "(e.g. 'CABLE Input').")
    parser.add_argument("--csv", default="mrp40_results.csv",
                        help="Output CSV path (default mrp40_results.csv).")

    parser.add_argument("--samples", type=int, default=3,
                        help="Samples per cell (default 3).")
    parser.add_argument("--snr-levels", type=float, nargs="+",
                        default=[25.0, 6.0],
                        help="SNR levels in dB (default 25 6).")
    parser.add_argument("--wpm-levels", type=int, nargs="+",
                        default=[15, 25],
                        help="WPM levels (default 15 25).")
    parser.add_argument("--key-types", nargs="+",
                        default=["straight", "bug", "paddle", "cootie"],
                        help="Key types (default all four).")
    parser.add_argument("--min-chars", type=int, default=60)
    parser.add_argument("--max-chars", type=int, default=100)

    parser.add_argument("--flush-margin-sec", type=float, default=1.5,
                        help="Wait this long after playback before reading "
                             "the log (default 1.5 s).")
    parser.add_argument("--inter-sample-silence-sec", type=float, default=2.0,
                        help="Silence played between samples so MRP40 "
                             "flushes any trailing decode (default 2.0 s).")
    parser.add_argument("--warmup-sec", type=float, default=3.0,
                        help="Silence at start to let MRP40 settle "
                             "(default 3 s).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Generate audio and print the plan, but do "
                             "not play or read the log.")
    args = parser.parse_args()

    try:
        import sounddevice as sd  # noqa: F401
    except ImportError:
        print("Error: sounddevice is required.  pip install sounddevice",
              file=sys.stderr)
        return 1

    if args.list_devices:
        print(sd.query_devices())
        return 0

    if not args.output_device:
        print("Error: --output-device is required.", file=sys.stderr)
        return 1
    if not args.mrp40_log and not args.dry_run:
        print("Error: --mrp40-log is required (unless --dry-run).",
              file=sys.stderr)
        return 1

    out_idx = resolve_output_device(args.output_device, sd)
    out_name = sd.query_devices(out_idx)["name"]
    print(f"Audio output device: [{out_idx}] {out_name}")
    mrp40_log = Path(args.mrp40_log) if args.mrp40_log else None
    if mrp40_log is not None:
        if not mrp40_log.exists():
            print(f"Warning: log file does not exist yet: {mrp40_log}")
            print("  Creating empty file. Confirm MRP40 writes to this path.")
            mrp40_log.parent.mkdir(parents=True, exist_ok=True)
            mrp40_log.touch()
        print(f"MRP40 log file:      {mrp40_log}")

    snr_levels = list(args.snr_levels)
    wpm_levels = list(args.wpm_levels)
    key_types = list(args.key_types)
    n = args.samples
    sample_rate = 16000
    total_cells = len(snr_levels) * len(wpm_levels) * len(key_types)
    total_samples = total_cells * n

    print()
    print(f"Grid: {len(snr_levels)} SNR x {len(wpm_levels)} WPM x "
          f"{len(key_types)} keys x {n} samples = {total_samples} samples")
    print(f"Char count per sample: [{args.min_chars}, {args.max_chars}]")

    # ------------------------------------------------------------------
    # Pre-generate all audio so we know the exact total duration before
    # committing real time to it (and so CER comparison uses identical
    # inputs if the user re-runs with a different sample_rate later).
    # ------------------------------------------------------------------
    samples: List[Tuple[float, int, str, int, np.ndarray, str, dict]] = []
    print("\nPre-generating audio ...", flush=True)
    t_gen = time.time()
    for snr in snr_levels:
        for wpm in wpm_levels:
            for kt in key_types:
                seed = 60000 + int((snr + 20) * 1000) + wpm * 10 + key_types.index(kt)
                rng = np.random.default_rng(seed)
                mc = make_config(snr, wpm, kt, args.min_chars, args.max_chars)
                for si in range(n):
                    audio, text, meta = generate_sample(mc, rng=rng)
                    samples.append((snr, wpm, kt, si, audio, text, meta))
    t_gen = time.time() - t_gen
    total_audio_sec = sum(float(s[6]["duration_sec"]) for s in samples)
    per_sample_overhead = args.flush_margin_sec + args.inter_sample_silence_sec
    est_wall_sec = total_audio_sec + per_sample_overhead * total_samples + args.warmup_sec

    print(f"Generated {len(samples)} samples in {t_gen:.1f}s.")
    print(f"Total audio duration: {total_audio_sec/60:.1f} min "
          f"({total_audio_sec:.0f} s)")
    print(f"Est. wall clock (1x real-time + overhead): "
          f"{est_wall_sec/60:.1f} min")

    if args.dry_run:
        print("\nDry run — not playing audio.")
        return 0

    # ------------------------------------------------------------------
    # Execute benchmark.
    # ------------------------------------------------------------------
    csv_fh = open(args.csv, "w", newline="", encoding="utf-8")
    fields = ["snr_db", "wpm", "key_type", "sample_idx",
              "ref_text", "hyp_text", "cer", "ref_len", "audio_sec"]
    writer = csv.DictWriter(csv_fh, fieldnames=fields)
    writer.writeheader()

    if args.warmup_sec > 0:
        print(f"\nWarm-up silence ({args.warmup_sec:.1f} s)...", flush=True)
        silence = np.zeros(int(args.warmup_sec * sample_rate), dtype=np.float32)
        sd.play(silence, samplerate=sample_rate, device=out_idx)
        sd.wait()

    results: Dict[Tuple, List[float]] = {}
    t_start = time.time()
    for idx, (snr, wpm, kt, si, audio, text, meta) in enumerate(samples, start=1):
        audio_sec = float(meta["duration_sec"])

        # Mark the log position before the sample plays.
        try:
            log_pos = mrp40_log.stat().st_size
        except OSError:
            log_pos = 0

        # Play sample.
        sd.play(audio.astype(np.float32), samplerate=sample_rate, device=out_idx)
        sd.wait()

        # Flush margin.
        time.sleep(args.flush_margin_sec)

        # Read new log content.
        try:
            with open(mrp40_log, "r", encoding="utf-8", errors="replace") as f:
                f.seek(log_pos)
                raw = f.read()
        except OSError as e:
            print(f"  [{idx:3d}/{total_samples}] log read fail: {e}",
                  file=sys.stderr)
            raw = ""

        hyp = clean_mrp40_output(raw)
        cer = compute_cer(hyp, text)
        results.setdefault((snr, wpm, kt), []).append(cer)

        writer.writerow({
            "snr_db": snr, "wpm": wpm, "key_type": kt,
            "sample_idx": si,
            "ref_text": text, "hyp_text": hyp,
            "cer": f"{cer:.4f}", "ref_len": len(text.strip()),
            "audio_sec": f"{audio_sec:.1f}",
        })
        csv_fh.flush()

        elapsed = time.time() - t_start
        eta = elapsed / idx * (total_samples - idx)
        print(f"  [{idx:3d}/{total_samples}] "
              f"SNR={snr:>4.0f} WPM={wpm:>2d} {kt:>8s} #{si}  "
              f"audio={audio_sec:5.1f}s  CER={cer:5.1%}  "
              f"(ETA {eta/60:.0f}m)", flush=True)

        # Silence gap so MRP40 flushes before next sample.
        if args.inter_sample_silence_sec > 0 and idx < total_samples:
            silence = np.zeros(
                int(args.inter_sample_silence_sec * sample_rate),
                dtype=np.float32,
            )
            sd.play(silence, samplerate=sample_rate, device=out_idx)
            sd.wait()

    wall_sec = time.time() - t_start

    # ------------------------------------------------------------------
    # Summary.
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("MRP40 Benchmark Results")
    print("=" * 72)
    print(f"\n{'SNR':>4s} {'WPM':>4s}", end="")
    for kt in key_types:
        print(f" {kt:>10s}", end="")
    print(f"    Avg")
    print("-" * (10 + 11 * len(key_types) + 7))

    for snr in snr_levels:
        for wpm in wpm_levels:
            print(f"{int(snr):4d} {wpm:4d}", end="")
            row = []
            for kt in key_types:
                cell = results.get((snr, wpm, kt), [])
                row.extend(cell)
                mean = float(np.mean(cell)) if cell else float("nan")
                print(f" {mean:9.1%}", end="")
            mean_row = float(np.mean(row)) if row else float("nan")
            print(f"  {mean_row:5.1%}")

    all_cers = [c for v in results.values() for c in v]
    rate = total_audio_sec / max(wall_sec, 1e-9)
    print(f"\nOverall CER:     {float(np.mean(all_cers)):.1%}")
    print(f"Total audio:     {total_audio_sec/60:.1f} min")
    print(f"Wall clock:      {wall_sec/60:.1f} min")
    print(f"Effective rate:  {rate:.2f}x real-time")
    print(f"CSV saved to:    {args.csv}")
    csv_fh.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())

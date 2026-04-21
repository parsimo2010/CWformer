#!/usr/bin/env python3
"""
benchmark_random_sweep.py — Large-scale random CW-Former evaluation.

Generates samples with ALL parameters varied randomly per the 'full' training
scenario.  Logs every sample's parameters and CER to CSV for regression analysis.

Usage:
    python benchmark_random_sweep.py --n 5000 --device cuda
"""

from __future__ import annotations

import argparse
import csv
import sys
import time

import numpy as np
import torch

from config import create_default_config
from metrics import compute_cer
from morse_generator import generate_sample
from neural_decoder.inference_cwformer import CWFormerStreamingDecoder


# ---------------------------------------------------------------------------
# CSV fields
# ---------------------------------------------------------------------------

CSV_FIELDS = [
    "sample_idx",
    # Signal parameters
    "wpm", "snr_db", "key_type", "tone_freq_hz",
    # Timing parameters
    "dah_dit_ratio", "ics_factor", "iws_factor", "timing_jitter",
    # Augmentation states
    "agc_depth_db", "qsb_depth_db",
    "qrm", "qrm_count", "qrn",
    "bandpass", "bandpass_bw",
    "farnsworth_stretch",
    "hf_noise", "multi_op",
    # Audio / decode info
    "audio_sec", "ref_len", "n_windows",
    # Results
    "ref_text", "hyp_text", "cer",
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Large-scale random CW-Former evaluation")
    parser.add_argument("--checkpoint",
                        default="checkpoints_cwformer_full/best_model.pt")
    parser.add_argument("--device",
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--n", type=int, default=5000,
                        help="Number of samples to evaluate")
    parser.add_argument("--csv", default="benchmark_random_5k.csv",
                        help="Output CSV path")
    parser.add_argument("--seed", type=int, default=12345,
                        help="RNG seed for reproducibility")
    parser.add_argument("--chunk-ms", type=int, default=500, dest="chunk_ms",
                        help="Streaming chunk size in milliseconds")
    args = parser.parse_args()

    # Full-scenario config: all augmentations randomly sampled
    cfg = create_default_config("full")
    mc = cfg.morse
    # Reasonable char range for manageable audio length across 5-50 WPM
    mc.min_chars = 40
    mc.max_chars = 150

    dec = CWFormerStreamingDecoder(
        checkpoint=args.checkpoint,
        chunk_ms=args.chunk_ms,
        device=args.device,
    )

    csv_fh = open(args.csv, "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(csv_fh, fieldnames=CSV_FIELDS)
    writer.writeheader()

    rng = np.random.default_rng(args.seed)
    n = args.n
    t0 = time.time()
    total_cer = 0.0
    completed = 0
    skipped = 0

    print(f"Running {n} random evaluations (full scenario, streaming greedy decode)")
    print(f"  WPM: {mc.min_wpm}-{mc.max_wpm}, SNR: {mc.min_snr_db}-{mc.max_snr_db} dB")
    print(f"  Chunk: {args.chunk_ms}ms")
    print(f"  Output: {args.csv}")
    print()

    for i in range(n):
        try:
            audio, text, meta = generate_sample(mc, rng=rng)
        except Exception as e:
            skipped += 1
            continue

        hyp = dec.decode_audio(audio)
        cer = compute_cer(hyp, text)

        # Compute chunk count for streaming
        chunk_samples = int(args.chunk_ms * dec.sample_rate / 1000)
        n_chunks = max(1, (len(audio) + chunk_samples - 1) // chunk_samples)

        writer.writerow({
            "sample_idx": i,
            "wpm": f"{meta['wpm']:.1f}",
            "snr_db": f"{meta['snr_db']:.1f}",
            "key_type": meta["key_type"],
            "tone_freq_hz": f"{meta['base_frequency_hz']:.0f}",
            "dah_dit_ratio": f"{meta['dah_dit_ratio']:.3f}",
            "ics_factor": f"{meta['ics_factor']:.3f}",
            "iws_factor": f"{meta['iws_factor']:.3f}",
            "timing_jitter": f"{meta['timing_jitter']:.3f}",
            "agc_depth_db": f"{meta['agc_depth_db']:.1f}",
            "qsb_depth_db": f"{meta['qsb_depth_db']:.1f}",
            "qrm": int(meta["qrm"]),
            "qrm_count": meta["qrm_count"],
            "qrn": int(meta["qrn"]),
            "bandpass": int(meta["bandpass"]),
            "bandpass_bw": f"{meta['bandpass_bw']:.0f}",
            "farnsworth_stretch": f"{meta['farnsworth_stretch']:.2f}",
            "hf_noise": int(meta["hf_noise"]),
            "multi_op": int(meta["multi_op"]),
            "audio_sec": f"{meta['duration_sec']:.1f}",
            "ref_len": len(text.strip()),
            "n_windows": n_chunks,
            "ref_text": text.strip(),
            "hyp_text": hyp.strip(),
            "cer": f"{cer:.4f}",
        })

        total_cer += cer
        completed += 1

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            rate = completed / elapsed
            eta = (n - i - 1) / rate if rate > 0 else 0
            running_cer = total_cer / completed
            csv_fh.flush()
            print(f"  [{i+1:5d}/{n}]  "
                  f"running CER={running_cer:5.1%}  "
                  f"{rate:.1f} samples/s  "
                  f"ETA {eta/60:.0f}m  "
                  f"(skipped {skipped})", flush=True)

    csv_fh.close()

    elapsed = time.time() - t0
    mean_cer = total_cer / completed if completed > 0 else 0

    print(f"\n{'=' * 60}")
    print(f"Done: {completed} evaluated, {skipped} skipped")
    print(f"Overall CER: {mean_cer:.1%}")
    print(f"Time: {elapsed/60:.1f} minutes ({completed/elapsed:.1f} samples/s)")
    print(f"CSV: {args.csv}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()

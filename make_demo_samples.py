#!/usr/bin/env python3
"""
make_demo_samples.py — Produce a small set of demo WAV files for human
listening tests.

Generates four 2-minute samples under demo_samples/:
  * demo_clean_paddle_snr25dB.wav  — paddle, 500 Hz BPF, +25 dB SNR
  * demo_clean_paddle_snr06dB.wav  — paddle, 500 Hz BPF,  +6 dB SNR
  * demo_clean_paddle_snr-5dB.wav  — paddle, 500 Hz BPF,  -5 dB SNR
  * demo_challenging_straight_snr06dB.wav — straight key, +6 dB SNR,
       moderate jitter + drift + AGC + QSB + QRM + QRN

Each WAV has an adjacent .txt with the ground-truth text.

Usage:
    python make_demo_samples.py
    python make_demo_samples.py --out demo_samples
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf

from config import MorseConfig
from morse_generator import generate_sample


SAMPLE_RATE = 16000
WPM = 20  # 2 min at 20 WPM ~ 200 chars


def _clean_paddle_cfg(snr_db: float) -> MorseConfig:
    """Paddle, 500 Hz filter, essentially no jitter, no augmentations."""
    mc = MorseConfig()
    # Nominal timing — dah/dit=3, ics=1, iws=1, pinned.
    mc.dah_dit_ratio_min = 3.0
    mc.dah_dit_ratio_max = 3.0
    mc.ics_factor_min = 1.0
    mc.ics_factor_max = 1.0
    mc.iws_factor_min = 1.0
    mc.iws_factor_max = 1.0
    mc.timing_jitter = 0.0
    mc.timing_jitter_max = 0.02   # negligible
    mc.tone_drift = 0.0
    mc.rise_time_ms_min = 5.0
    mc.rise_time_ms_max = 5.0
    # Fixed 500 Hz, order 6 bandpass — the only "augmentation".
    mc.bandpass_probability = 1.0
    mc.bandpass_bw_min = 500.0
    mc.bandpass_bw_max = 500.0
    mc.bandpass_order_min = 6
    mc.bandpass_order_max = 6
    # Everything else off.
    mc.agc_probability = 0.0
    mc.qsb_probability = 0.0
    mc.qrm_probability = 0.0
    mc.qrn_probability = 0.0
    mc.hf_noise_probability = 0.0
    mc.farnsworth_probability = 0.0
    mc.multi_op_probability = 0.0
    mc.speed_drift_max = 0.0
    # Length & speed.
    mc.min_chars = 200
    mc.max_chars = 200
    mc.min_wpm = WPM
    mc.max_wpm = WPM
    # Paddle only.
    mc.key_type_weights = (0.0, 0.0, 1.0, 0.0)
    # SNR pinned.
    mc.min_snr_db = snr_db
    mc.max_snr_db = snr_db
    return mc


def _challenging_straight_cfg() -> MorseConfig:
    """Straight key at +6 dB SNR with moderate jitter, drift, and QRM/QRN/QSB/AGC."""
    mc = MorseConfig()
    # Straight-key operator with a noticeable but not extreme fist.
    mc.dah_dit_ratio_min = 2.5
    mc.dah_dit_ratio_max = 3.5
    mc.ics_factor_min = 0.8
    mc.ics_factor_max = 1.5
    mc.iws_factor_min = 0.8
    mc.iws_factor_max = 1.7
    mc.timing_jitter = 0.0
    mc.timing_jitter_max = 0.18    # moderate jitter
    mc.tone_drift = 5.0
    mc.rise_time_ms_min = 4.0
    mc.rise_time_ms_max = 8.0
    mc.speed_drift_max = 0.12      # ~12% speed drift across the sample
    # 500 Hz BPF as requested.
    mc.bandpass_probability = 1.0
    mc.bandpass_bw_min = 500.0
    mc.bandpass_bw_max = 500.0
    mc.bandpass_order_min = 6
    mc.bandpass_order_max = 6
    # AGC always on, 12 dB pumping.
    mc.agc_probability = 1.0
    mc.agc_depth_db_min = 10.0
    mc.agc_depth_db_max = 12.0
    # QSB (slow fading), noticeable.
    mc.qsb_probability = 0.8
    mc.qsb_depth_db_min = 8.0
    mc.qsb_depth_db_max = 12.0
    # One or two adjacent interferers, audible but not overwhelming.
    mc.qrm_probability = 1.0
    mc.qrm_count_min = 1
    mc.qrm_count_max = 2
    mc.qrm_amplitude_min = 0.3
    mc.qrm_amplitude_max = 0.45
    # Atmospheric crashes at 2-4 per second.
    mc.qrn_probability = 1.0
    mc.qrn_rate_min = 2.0
    mc.qrn_rate_max = 4.0
    # Keep HF noise / Farnsworth / multi-op off to isolate the stated
    # augmentation set.
    mc.hf_noise_probability = 0.0
    mc.farnsworth_probability = 0.0
    mc.multi_op_probability = 0.0
    # Length & speed.
    mc.min_chars = 200
    mc.max_chars = 200
    mc.min_wpm = WPM
    mc.max_wpm = WPM
    # Straight key only.
    mc.key_type_weights = (1.0, 0.0, 0.0, 0.0)
    mc.min_snr_db = 6.0
    mc.max_snr_db = 6.0
    return mc


def _challenging_lowsnr_cfg() -> MorseConfig:
    """Straight key at -10 dB SNR with aggressive jitter/drift/AGC/QSB/QRN.

    No QRM (no other signal on frequency) — the difficulty comes from
    deep noise, deep QSB fades, frequent static crashes, and a shaky
    operator fist.
    """
    mc = MorseConfig()
    # Straight-key operator, shakier fist than the +6 dB challenging sample.
    mc.dah_dit_ratio_min = 2.5
    mc.dah_dit_ratio_max = 3.5
    mc.ics_factor_min = 0.8
    mc.ics_factor_max = 1.5
    mc.iws_factor_min = 0.8
    mc.iws_factor_max = 1.7
    mc.timing_jitter = 0.0
    mc.timing_jitter_max = 0.20    # high jitter
    mc.tone_drift = 5.0
    mc.rise_time_ms_min = 4.0
    mc.rise_time_ms_max = 8.0
    mc.speed_drift_max = 0.15      # 15% speed drift
    # 500 Hz BPF.
    mc.bandpass_probability = 1.0
    mc.bandpass_bw_min = 500.0
    mc.bandpass_bw_max = 500.0
    mc.bandpass_order_min = 6
    mc.bandpass_order_max = 6
    # AGC, hard pumping.
    mc.agc_probability = 1.0
    mc.agc_depth_db_min = 10.0
    mc.agc_depth_db_max = 14.0
    # Deep QSB — signal nearly disappears at fade minima.
    mc.qsb_probability = 0.9
    mc.qsb_depth_db_min = 12.0
    mc.qsb_depth_db_max = 16.0
    # No QRM — stated requirement.
    mc.qrm_probability = 0.0
    # Heavier atmospheric crashes.
    mc.qrn_probability = 1.0
    mc.qrn_rate_min = 3.0
    mc.qrn_rate_max = 6.0
    # No HF noise / Farnsworth / multi-op.
    mc.hf_noise_probability = 0.0
    mc.farnsworth_probability = 0.0
    mc.multi_op_probability = 0.0
    # Length & speed.
    mc.min_chars = 200
    mc.max_chars = 200
    mc.min_wpm = WPM
    mc.max_wpm = WPM
    # Straight key only.
    mc.key_type_weights = (1.0, 0.0, 0.0, 0.0)
    # Low SNR.
    mc.min_snr_db = -10.0
    mc.max_snr_db = -10.0
    return mc


def _write_sample(out_dir: Path, stem: str, mc: MorseConfig, seed: int) -> None:
    rng = np.random.default_rng(seed)
    audio, text, meta = generate_sample(mc, rng=rng)
    wav_path = out_dir / f"{stem}.wav"
    txt_path = out_dir / f"{stem}.txt"
    sf.write(str(wav_path), audio, SAMPLE_RATE)
    txt_path.write_text(text, encoding="utf-8")
    print(
        f"  {wav_path.name}  "
        f"{meta['duration_sec']:6.1f}s  "
        f"{meta['wpm']:.0f} WPM  "
        f"SNR={meta['snr_db']:+.0f} dB  "
        f"{len(text)} chars  "
        f"key={meta['key_type']}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--out", default="demo_samples",
                        help="Output directory (default: demo_samples).")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Writing to: {out_dir.resolve()}\n")

    print("Clean paddle, 500 Hz BPF, SNR sweep:")
    for snr in (25.0, 6.0, -5.0):
        stem = f"demo_clean_paddle_snr{int(snr):+d}dB".replace("+", "+")
        _write_sample(out_dir, stem, _clean_paddle_cfg(snr),
                      seed=1000 + int(snr + 10))

    print("\nChallenging straight key, 500 Hz BPF, SNR=+6 dB, "
          "moderate jitter/drift/AGC/QSB/QRM/QRN:")
    _write_sample(out_dir, "demo_challenging_straight_snr+6dB",
                  _challenging_straight_cfg(), seed=2000)

    print("\nChallenging straight key, 500 Hz BPF, SNR=-10 dB, "
          "heavy jitter/drift/AGC/deep QSB/QRN (no QRM):")
    _write_sample(out_dir, "demo_challenging_straight_snr-10dB",
                  _challenging_lowsnr_cfg(), seed=3000)

    print("\nDone. Each .wav has an adjacent .txt with ground truth.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

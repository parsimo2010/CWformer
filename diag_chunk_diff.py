"""Chunk-by-chunk log-prob diff: PyTorch streaming vs ONNX FP32 streaming.

Answers the question "is ONNX/PyTorch divergence a gradual numerical drift
or a sudden jump?" by feeding the same audio through both decoders and
diffing their per-chunk log_probs.

Trajectory interpretation:
  - Large diff from chunk 0  -> trace bug (RoPE, shape, etc.)
  - Small flat diff          -> both agree; text divergence is elsewhere
  - Smoothly growing diff    -> numerical compounding (noise sensitivity)
  - Sudden jump at chunk N   -> specific input triggers a bug

Usage:
    python diag_chunk_diff.py [audio_file]

Defaults to recordings/web1.wav.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from neural_decoder.inference_cwformer import (
    CWFormerStreamingDecoder,
    _peak_normalize,
)
from deploy.inference_onnx import CWFormerStreamingONNX, load_audio

CHECKPOINT = "checkpoints_v2/best_model.pt"
ONNX_MODEL = "deploy/cwformer_streaming_fp32.onnx"


def main() -> None:
    audio_path = sys.argv[1] if len(sys.argv) > 1 else "recordings/web1.wav"

    print(f"PyTorch : {CHECKPOINT}")
    print(f"ONNX    : {ONNX_MODEL}")
    print(f"Audio   : {audio_path}")
    print()

    pt = CWFormerStreamingDecoder(CHECKPOINT)
    onnx = CWFormerStreamingONNX(ONNX_MODEL)

    # Sanity: both decoders must agree on chunk size. decode_audio() peak-
    # normalizes on entry; we do it here once so both feeds see identical
    # samples. feed_audio() itself does NOT normalize.
    audio = load_audio(audio_path, pt.sample_rate)
    audio = _peak_normalize(audio, target_peak=0.7)

    assert pt._chunk_samples == onnx._chunk_samples, "chunk size mismatch"
    chunk_samples = pt._chunk_samples

    print(f"Chunk   : {chunk_samples} samples ({chunk_samples/16:.0f} ms)")
    print(f"Audio   : {len(audio)} samples ({len(audio)/16000:.2f} s)")
    print()

    pt.reset()
    onnx.reset()

    rows = []
    pos = 0
    chunk_idx = 0

    while pos + chunk_samples <= len(audio):
        chunk = audio[pos:pos + chunk_samples]

        pt_before = len(pt._all_log_probs)
        onnx_before = len(onnx._all_log_probs)

        pt.feed_audio(chunk)
        onnx.feed_audio(chunk)

        if (len(pt._all_log_probs) > pt_before
                and len(onnx._all_log_probs) > onnx_before):
            pt_lp = pt._all_log_probs[-1]
            onnx_lp = onnx._all_log_probs[-1]
            pt_np = pt_lp.numpy() if hasattr(pt_lp, "numpy") else pt_lp

            if pt_np.shape != onnx_lp.shape:
                print(f"  chunk {chunk_idx:3d}: SHAPE MISMATCH "
                      f"pt={pt_np.shape} onnx={onnx_lp.shape}")
                rows.append((chunk_idx, float("nan"), float("nan"),
                             pt_np.shape, onnx_lp.shape))
            else:
                max_diff = float(np.max(np.abs(pt_np - onnx_lp)))
                mean_diff = float(np.mean(np.abs(pt_np - onnx_lp)))
                rows.append((chunk_idx, max_diff, mean_diff,
                             pt_np.shape, onnx_lp.shape))

        pos += chunk_samples
        chunk_idx += 1

    # Print a condensed table
    print(f"{'chunk':>5}  {'max |diff|':>12}  {'mean |diff|':>12}  shape")
    print("-" * 60)
    for idx, mx, mn, sh, _ in rows:
        print(f"{idx:>5}  {mx:>12.6f}  {mn:>12.6f}  {sh}")

    # Verdict
    valid = [(idx, mx) for idx, mx, _, _, _ in rows
             if not np.isnan(mx)]
    if not valid:
        print("\n[error] no comparable chunks produced.")
        return

    first_diff = valid[0][1]
    max_diff = max(d for _, d in valid)
    last_diff = valid[-1][1]

    print()
    print(f"First chunk diff : {first_diff:.6f}")
    print(f"Last chunk diff  : {last_diff:.6f}")
    print(f"Max over all     : {max_diff:.6f}")

    step_jumps = [
        (i, valid[i][0], valid[i - 1][1], valid[i][1])
        for i in range(1, len(valid))
        if abs(valid[i][1] - valid[i - 1][1]) > 0.05
    ]

    print()
    if first_diff > 0.01:
        print("VERDICT: large diff from chunk 0. Likely a trace bug — some")
        print("         op produces materially different output on the very")
        print("         first call. Worth investigating the graph.")
    elif max_diff < 0.01:
        print("VERDICT: consistently small diff (< 0.01). PyTorch and ONNX")
        print("         agree at the log-prob level. Any text-level divergence")
        print("         is below the CTC noise floor — nothing to fix here.")
    elif step_jumps:
        print(f"VERDICT: {len(step_jumps)} sudden jump(s). First at chunk "
              f"{step_jumps[0][1]}: {step_jumps[0][2]:.4f} -> "
              f"{step_jumps[0][3]:.4f}.")
        print("         Something about that chunk's input triggers divergent")
        print("         behavior. Worth investigating what's special there.")
    else:
        print("VERDICT: smoothly growing diff — numerical compounding from")
        print("         FP32 matmul differences between PyTorch and ORT. On")
        print("         clean audio the top CTC class wins anyway; on noisy")
        print("         audio near-ties get flipped. This is intrinsic to")
        print("         running two FP32 backends and not fixable without")
        print("         moving to FP64 (not realistic for ONNX deployment).")


if __name__ == "__main__":
    main()

"""Mel-only diff: PyTorch MelFrontend vs numpy MelComputer.

Isolates the mel frontend from the rest of the pipeline. Feeds identical
audio chunks through both implementations (``torch.stft`` vs ``numpy.rfft
+ manual Hann window``) and reports per-chunk divergence of both the
emitted log-mel features AND the STFT overlap buffer that carries state
between chunks.

Answers: does the 8+ log-prob gap we see after chunk 0 originate upstream
of the model graph (in the mel frontend itself), or is it purely a model-
graph issue?

Usage:
    python tests/diagnostic/diag_mel_diff.py [audio_file]

Edit CHECKPOINT / ONNX_MODEL below to point at the files you want to diff.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from neural_decoder.inference_cwformer import (
    CWFormerStreamingDecoder,
    _peak_normalize,
)
from deploy.inference_onnx import CWFormerStreamingONNX, load_audio

CHECKPOINT = "checkpoints_v2/best_model.pt"
ONNX_MODEL = "deploy/cwformer_streaming_fp32.onnx"


def main() -> None:
    audio_path = sys.argv[1] if len(sys.argv) > 1 else "recordings/web1.wav"

    pt = CWFormerStreamingDecoder(CHECKPOINT)
    onnx = CWFormerStreamingONNX(ONNX_MODEL)

    assert pt._chunk_samples == onnx._chunk_samples, "chunk size mismatch"
    chunk_samples = pt._chunk_samples

    audio = load_audio(audio_path, pt.sample_rate)
    audio = _peak_normalize(audio, target_peak=0.7)

    print(f"PyTorch mel : {CHECKPOINT}")
    print(f"ONNX    mel : {ONNX_MODEL}")
    print(f"Audio       : {audio_path}")
    print(f"Chunk       : {chunk_samples} samples ({chunk_samples/16:.0f} ms)")
    print(f"Audio       : {len(audio)} samples ({len(audio)/16000:.2f} s)")
    print(f"n_fft       : {onnx.mel.n_fft}, hop : {onnx.mel.hop}")
    print()

    # Compare static pieces first: window function and mel filterbank.
    # If these disagree, every mel value diverges regardless of input.
    pt_window = pt._model.mel_frontend.window.numpy()
    onnx_window = onnx.mel.window
    pt_fb = pt._model.mel_frontend.mel_basis.numpy()
    onnx_fb = onnx.mel.mel_basis

    win_max = float(np.max(np.abs(pt_window - onnx_window)))
    fb_max = float(np.max(np.abs(pt_fb - onnx_fb)))
    print(f"Hann window max |diff|    : {win_max:.6e}")
    print(f"Mel filterbank max |diff| : {fb_max:.6e}")
    print()

    # Streaming comparison
    pt_buf = None    # (B, N) torch.Tensor
    onnx_buf = None  # (N,)   np.ndarray

    rows = []
    pos = 0
    chunk_idx = 0
    while pos + chunk_samples <= len(audio):
        chunk_np = audio[pos:pos + chunk_samples]
        chunk_t = torch.from_numpy(chunk_np).unsqueeze(0)

        pt_mel, pt_buf = pt._model.mel_frontend.compute_streaming(
            chunk_t, pt_buf,
        )
        onnx_mel, onnx_buf = onnx.mel.compute_streaming(
            chunk_np, onnx_buf,
        )

        pt_mel_np = pt_mel[0].numpy()
        onnx_mel_np = onnx_mel[0]

        # Per-chunk mel diff
        if pt_mel_np.shape == onnx_mel_np.shape:
            mel_diff = pt_mel_np - onnx_mel_np
            mel_max = float(np.max(np.abs(mel_diff)))
            mel_mean = float(np.mean(np.abs(mel_diff)))
            # Locate the spike: what log-mel value sits at the max-diff
            # position? log(mel + 1e-6) is ~-13.8 at silence (noise
            # floor), 0 to 5 at active signal. If spikes cluster near
            # -13.8, silence-at-noise-floor hypothesis holds.
            flat_idx = int(np.argmax(np.abs(mel_diff)))
            t_idx, f_idx = np.unravel_index(flat_idx, mel_diff.shape)
            spike_pt = float(pt_mel_np[t_idx, f_idx])
            spike_onnx = float(onnx_mel_np[t_idx, f_idx])
        else:
            mel_max = float("nan")
            mel_mean = float("nan")
            t_idx = f_idx = -1
            spike_pt = spike_onnx = float("nan")

        # Per-chunk stft buffer diff (the state that drifts between calls)
        pt_buf_np = pt_buf[0].numpy() if pt_buf is not None else None
        if pt_buf_np is not None and onnx_buf is not None \
                and pt_buf_np.shape == onnx_buf.shape:
            buf_max = float(np.max(np.abs(pt_buf_np - onnx_buf)))
        else:
            buf_max = float("nan")

        rows.append((
            chunk_idx, pt_mel_np.shape,
            mel_max, mel_mean, buf_max,
            t_idx, f_idx, spike_pt, spike_onnx,
        ))

        pos += chunk_samples
        chunk_idx += 1

    # Table.  Spike columns show the log-mel values at the argmax-diff
    # position — lets you see whether spikes live at the noise floor
    # (~-13.8, silence) or at active-signal magnitudes (~0 to 5).
    print(f"{'chunk':>5}  {'mel_shape':>12}  "
          f"{'mel max|diff|':>15}  {'mel mean|diff|':>15}  "
          f"{'buf max|diff|':>15}  "
          f"{'t':>3} {'f':>3}  {'pt@spike':>10} {'onnx@spike':>10}")
    print("-" * 110)
    for idx, shape, mx, mn, buf, t_i, f_i, sp_pt, sp_ox in rows:
        shape_str = str(shape)
        print(f"{idx:>5}  {shape_str:>12}  "
              f"{mx:>15.6e}  {mn:>15.6e}  {buf:>15.6e}  "
              f"{t_i:>3} {f_i:>3}  {sp_pt:>10.4f} {sp_ox:>10.4f}")

    # Classify spikes: silence (log-mel <= -10) vs mid (-10..-5) vs
    # active signal (> -5). If spikes are overwhelmingly at silence,
    # the noise-floor FP32 hypothesis holds; otherwise the spike cause
    # lives somewhere else.
    spike_silence = 0
    spike_mid = 0
    spike_active = 0
    for _, _, mx, _, _, _, _, sp_pt, _ in rows:
        if np.isnan(mx):
            continue
        val = min(sp_pt, 0)  # use the more-negative of the pair as floor indicator
        if sp_pt <= -10:
            spike_silence += 1
        elif sp_pt <= -5:
            spike_mid += 1
        else:
            spike_active += 1
    total = spike_silence + spike_mid + spike_active
    if total > 0:
        print()
        print(f"Spike location tally (pt log-mel at argmax-diff):")
        print(f"  silence (<= -10)  : {spike_silence:>4} "
              f"({100*spike_silence/total:.0f}%)")
        print(f"  mid (-10..-5)     : {spike_mid:>4} "
              f"({100*spike_mid/total:.0f}%)")
        print(f"  active (> -5)     : {spike_active:>4} "
              f"({100*spike_active/total:.0f}%)")
        if spike_silence / total > 0.8:
            print("  -> silence-dominant. FP32 noise-floor hypothesis fits.")
        elif spike_active / total > 0.5:
            print("  -> spikes are on ACTIVE signal. Hypothesis does NOT fit.")
        else:
            print("  -> mixed. Hypothesis partially fits.")

    # Summary
    valid = [(r[0], r[2], r[4]) for r in rows if not np.isnan(r[2])]
    if not valid:
        print("\n[error] no valid comparisons.")
        return

    first_mel = valid[0][1]
    last_mel = valid[-1][1]
    max_mel = max(v[1] for v in valid)
    max_buf = max(v[2] for v in valid if not np.isnan(v[2]))

    print()
    print(f"First chunk mel max |diff| : {first_mel:.6e}")
    print(f"Last chunk  mel max |diff| : {last_mel:.6e}")
    print(f"Max mel |diff| over all    : {max_mel:.6e}")
    print(f"Max buf |diff| over all    : {max_buf:.6e}")
    print()

    # Verdict
    if win_max > 1e-6 or fb_max > 1e-6:
        print("VERDICT: static tables (Hann window or mel filterbank) already")
        print("         disagree. Both implementations compute these from the")
        print("         same formulas — a non-trivial diff here points to a")
        print("         subtle bug (periodic-vs-symmetric window, off-by-one,")
        print("         float32 vs float64 casting). Fix this first.")
    elif max_mel < 1e-5:
        print("VERDICT: mels are bit-equivalent (< 1e-5). The mel frontend is")
        print("         NOT the source of the log-prob divergence. The bug is")
        print("         in the model graph — refine the diagnostic to diff")
        print("         intermediate encoder outputs.")
    elif max_mel < 1e-3:
        print("VERDICT: mels differ at ~1e-4. Plausible pure-numerics from")
        print("         torch.stft (MKL) vs numpy.rfft accumulation order.")
        print("         Compounds through 12 attention layers but probably")
        print("         accounts for only part of the observed log-prob drift.")
    else:
        print(f"VERDICT: mels differ by {max_mel:.3e} — above numerical noise.")
        print("         Real implementation divergence worth fixing. Check")
        print("         STFT centering, buffer carry logic, or mel filterbank")
        print("         normalization.")

    # Trajectory
    if len(valid) >= 2 and not np.isnan(max_buf):
        print()
        print("Trajectory check: does mel diff grow with chunk index?")
        early = np.mean([v[1] for v in valid[:len(valid)//3]])
        late = np.mean([v[1] for v in valid[-len(valid)//3:]])
        print(f"  early-third mean: {early:.6e}")
        print(f"  late-third mean : {late:.6e}")
        if late > early * 2:
            print("  -> drift is GROWING. Buffer state is compounding errors.")
        else:
            print("  -> drift is STEADY. Per-chunk numerical floor, not drift.")


if __name__ == "__main__":
    main()

# CWformer — Claude Reference Overview

## Project Intent & Goals

CWformer is a **causal streaming** neural Morse code (CW) decoder. It uses a
fully causal Conformer architecture (~19.5M params) with CTC loss that
processes audio left-to-right with no bidirectional attention.

**Why this exists:** Prior sliding-window decoders traded accuracy against
latency via window stitching. This project eliminates stitching entirely
by using a causal model with KV cache + conv-buffer state carry-forward
between processing chunks.

**Design philosophy:** Process audio causally (each frame sees only past
context), carry state between chunks via KV cache and conv buffers, and
emit characters as soon as greedy CTC decodes them. No commitment delay
is needed — causal CTC guarantees the previously-emitted prefix never
changes, so a character can be emitted the moment its frame's argmax is
stable.

**Target performance:** 15–40 WPM primary window, any key type (straight,
bug, paddle, cootie), SNR > 5–8 dB. Chunk-sized latency (~500 ms typical)
from audio to character emission. Desktop CPU/GPU deployment.

**Architecture model:** Causal Conformer, following NVIDIA FastConformer
(streaming mode) and Google streaming Conformer. CTC loss with greedy
decoding.

**See `PLAN.md`** for the original implementation plan from the bidirectional-
to-causal port. It is historical context only — where it disagrees with
current code, the code is correct.

---

## Architecture

```
Audio (16 kHz mono, streaming chunks)
  -> (Inference) peak-normalize to 0.7 on decode_audio entry (training is
     peak-normalized in morse_generator, so inference must match)
  -> MelFrontend: incremental log-mel spectrogram (40 bins, 200-1400 Hz, 25ms/10ms)
     + SpecAugment (training only)
  -> Causal ConvSubsampling: 2x time reduction (left-pad only in time)
     -> 50 fps (20ms per CTC frame)
  -> Causal ConformerEncoder: 12 blocks (d=256, 4 heads, conv kernel=63)
     - Self-attention: fully causal (is_causal=True during training, KV cache during inference)
     - Convolution: causal depthwise (left-pad only), LayerNorm after depthwise
     - Feed-forward: Macaron-style half-step
  -> Linear CTC head -> log_softmax -> incremental greedy decode -> text
```

**Conv kernel size**: 63 (1240 ms receptive field at 50 fps). Spans a
15 WPM inter-word space in a single block.

**Causal streaming properties:** Attention is fully causal (each frame
only attends to past frames), convolutions are left-padded only, and
inference uses KV cache + conv buffers for state continuity between
chunks. No window stitching. No commitment delay — causal CTC's prefix-
stability makes delayed emission unnecessary, and inference re-decodes
the accumulated log_probs each chunk (O(T²) total but correct: past
output never changes).

---

## File Map & Key Functions

### Infrastructure
- `config.py` — `MorseConfig`, `TrainingConfig`, `create_default_config(scenario)`. **sample_rate = 16000**.
- `vocab.py` — CTC vocabulary (52 classes). `encode(text)`, `decode(indices)`, `decode_ctc(log_probs)`.
- `metrics.py` — Shared Levenshtein / CER helpers (`compute_cer`, `levenshtein`, `per_position_errors`). Used by training, benchmarks, and `demo_samples/CER.py`.
- `morse_table.py` — ITU Morse code table + binary trie.
- `morse_generator.py` — Synthetic training data. `generate_sample(config)` -> `(audio_f32, text, metadata)`. All augmentations: AGC, QSB, QRM, QRN, bandpass, HF noise, key types, timing jitter, speed drift, input-gain.
- `qso_corpus.py` — `QSOCorpusGenerator` for realistic ham radio QSO text.

### neural_decoder/ — Causal CW-Former

#### Model
- `cwformer.py` — `CWFormer` (~19.5M params): MelFrontend -> Causal ConvSubsampling -> Causal ConformerEncoder -> CTC head.
  - `forward()` — Training: full sequence with causal attention. Input: `(audio, audio_lengths)` -> `(log_probs, output_lengths)`.
  - `forward_streaming(mel_chunk, state)` — Inference: single chunk with KV cache + conv buffers. Returns `(log_probs, new_state)`.
  - `ConvSubsampling` — Causal 2x time reduction (left-pad=2, right-pad=0 in time for both Conv2d layers). Streaming uses variable-length (1 or 2 frames) conv1 carry-over to stay aligned across chunks with odd padded length.
- `conformer.py` — Causal Conformer blocks.
  - `ConformerMHA` — Causal self-attention. Training: `is_causal=True` in SDPA. Inference: KV cache concatenation + causal mask within chunk. RoPE with position offset. RoPE tables auto-extend on demand for long sessions.
  - `ConvolutionModule` — Causal depthwise conv, left-pad only. `nn.LayerNorm` after the depthwise conv (per-frame statistics independent of batch composition and sequence length, matching streaming inference). Inference: conv buffer carry-forward of (kernel-1)=62 frames.
  - `ConformerEncoder` — Threads KV caches and conv buffers through 12 blocks. No padding mask: causality prevents valid frames from attending to padded positions, and LayerNorm is per-frame throughout.
- `rope.py` — Rotary Position Embeddings with `offset` parameter for KV cache positions. `_build_tables()` auto-extends the cos/sin table when a streaming session grows longer than the initial `max_len` (4096 default).
- `mel_frontend.py` — `MelFrontend` with `compute_streaming(audio_chunk, stft_buffer)` for incremental mel computation. STFT overlap buffer sized `n_fft - hop_length = 240` samples. `forward()` (training) pads audio by `n_fft//2` on both sides. `CWFormerStreamingDecoder.flush()` right-pads the final chunk by `n_fft//2` to match the training tail.

#### Training
- `dataset_audio.py` — `AudioDataset`: streaming IterableDataset. Samples per-sample WPM first, then constrains text length to fit within `max_audio_sec` (default 30 s in training).
- `train_cwformer.py` — Training loop. Micro-batch 8, effective batch 64 via gradient accumulation. Causal attention active during training. Supports optional streaming-mode validation and auto-curriculum progression (clean -> moderate -> full on CER plateau).

#### Inference
- `inference_cwformer.py` — `CWFormerStreamingDecoder`: chunk-based streaming with state carry-forward. No windows, no stitching. No commitment delay — emits characters as soon as greedy CTC decodes them (causal prefix stability). Methods: `feed_audio()`, `get_full_text()`, `flush()`, `decode_file()`, `decode_audio()`. `decode_audio()` peak-normalizes audio to 0.7 on entry to match the training distribution; `feed_audio()` does not normalize (caller owns live-audio gain). `max_cache_sec` caps the KV cache to the training-audio max (30 s by default).

### Deployment
- `quantize_cwformer.py` — Streaming ONNX export with state I/O (KV caches + conv buffers as explicit input/output tensors). INT8 dynamic quantization. Also saves `mel_basis.npy` and `mel_window.npy` bit-for-bit so deployment uses the exact tables the model was trained with.
- `deploy/inference_onnx.py` — `CWFormerStreamingONNX`: standalone ONNX-runtime inference with streaming state management. Supports file, device, and stdin input. `MelComputer.compute_streaming` matches the PyTorch path (variable-length carry-forward, left-pad on first chunk only). `flush()` right-pads by `n_fft // 2` and `decode_audio()` peak-normalizes to 0.7, both mirroring `CWFormerStreamingDecoder`. Live paths (`decode_live`, stdin) do not normalize — caller owns live-audio gain.

### Testing
- `tests/test_streaming_equivalence.py` — numerical equivalence check: runs the same audio through `CWFormer.forward()` (training path) and the chunk-by-chunk streaming path, then diffs every intermediate (mel, subsample, each encoder block, log_probs). Also has sub-step diffing inside block 0 (FF1/MHA/conv/FF2/final_norm) for pinpointing divergence. Runs up to three SDPA/device configurations; Run 3 (CPU, math SDPA) is the strictest and mirrors ONNX Runtime CPU numerics. Run after any change to the model or streaming logic.

### Benchmarking
- `benchmark_cwformer.py` — Structured benchmark across SNR, WPM, key types. Phase 1 (clean baseline grid with 100% bandpass on long audio), Phase 2 (single-augmentation marginal effect), Phase 3 (per-character-position error rate for context ramp-up).
- `benchmark_random_sweep.py` — Random parameter sweep benchmark using the `full` scenario distribution.

### Diagnostics (ad-hoc, not part of the test suite)
- `tests/diagnostic/diag_mel_diff.py` — Mel-frontend diff: torch.stft vs numpy MelComputer.
- `tests/diagnostic/diag_chunk_diff.py` — Per-chunk log-prob diff: PyTorch streaming vs ONNX streaming.
Both scripts hardcode checkpoint/ONNX paths at the top; edit before running.

---

## Curriculum Learning

Three training stages with progressively harder conditions. All stages
cover WPM 5–50 and use random input-gain augmentation (off at clean,
±6 dB at moderate, ±12 dB at full) so the model handles non-peak-
normalized recordings at inference time.

| Stage | SNR | WPM | AGC | QSB | Key Types (S/B/P/C) | Notable Augmentations |
|-------|-----|-----|-----|-----|---------------------|-----------------------|
| clean    | 15–40 dB  | 5–50 | 20% | 0%  | 20/20/60/0   | 10% Farnsworth, 50% bandpass, 15% HF noise |
| moderate | 5–35 dB   | 5–50 | 40% | 25% | 25/25/35/15  | 20% Farnsworth, 15% QRM, 15% QRN, 60% bandpass, 30% HF noise, ±6 dB input gain |
| full     | −5–30 dB  | 5–50 | 50% | 50% | 30/30/20/20  | 25% Farnsworth, 30% QRM, 25% QRN, 60% bandpass, 50% HF noise, 15% multi-op, ±12 dB input gain |

See `create_default_config()` in `config.py` for exact per-stage values.

---

## Performance Targets
- Primary window (15–40 WPM, any key type, SNR > 8 dB): < 5% CER goal.
- Extended envelope (5–50 WPM, low SNR down to −5 dB): graceful degradation.
- Latency: chunk size (e.g. 500 ms) + model compute (~30 ms).
- Real-time factor: < 0.1 (10x faster than real-time on desktop CPU).

---

## Things to Keep in Mind

1. **Sample rate is 16 kHz** — all audio is resampled to 16 kHz internally.
2. **2x subsampling gives 50 fps (20 ms per CTC frame)** — resolves dits up to 40+ WPM.
3. **Causal attention** — `is_causal=True` during training, KV cache + causal mask during inference. Never let a frame see future audio.
4. **Causal convolution** — left-pad only. Per-block receptive field = kernel × frame stride = 63 × 20 ms = 1240 ms. Inference maintains a (kernel-1)=62-frame conv buffer per layer.
5. **KV cache** — grows per chunk during inference. Trimmed to `max_cache_len` (1475 frames, ≈ 29.5 s; leaves 25-frame chunk headroom vs the 1500-frame training max so the post-concat attention window stays ≤ training max). Position offset tracks absolute position for RoPE; RoPE tables auto-extend if the session runs long.
6. **No commitment delay** — causal CTC's prefix-stability guarantees the emitted text never changes as more audio arrives, so characters emit as soon as greedy decode stabilizes. Don't reintroduce a delay without a specific reason.
7. **Boundary space tokens** — dataset wraps targets with `[space] + encode(text) + [space]`.
8. **Persistent worker RNG** — use `np.random.default_rng()` (OS entropy), not `worker_info.seed`.
9. **DataLoader tuning** — `persistent_workers=True`, `prefetch_factor=4`. Audio generation is the CPU bottleneck.
10. **Training uses full sequences** — no chunking during training. Inference chunks are for efficiency/latency, not for training. Streaming equivalence to full-forward is verified for in-distribution audio by `tests/test_streaming_equivalence.py`.
11. **Peak normalization** — `morse_generator.generate_sample()` peak-normalizes every training sample to `target_amplitude ∈ [0.5, 0.9]`, then applies random input-gain in dB (at moderate/full stages). `CWFormerStreamingDecoder.decode_audio()` peak-normalizes input to 0.7 to match. The live `feed_audio()` path does NOT normalize (caller is responsible).
12. **Greedy decode only** — no beam search or LM. Both were tested and didn't help; they add latency without CER gain.
13. **ONNX state I/O** — per-layer naming: `kv_k_layer{i}`, `kv_v_layer{i}`, `conv_buf_layer{i}` for each of 12 layers (36 tensors) + `sub_buf1`, `sub_buf2`, `pos_offset`. All dynamic over the time dimension.

---

## Implementation Phases

Work through these in order if you need to touch the stack end-to-end. Each phase is testable independently:

1. **Core model** — `rope.py` (offset), `conformer.py` (causal attn + conv + state), `cwformer.py` (causal subsampling + `forward_streaming`). Verify with `tests/test_streaming_equivalence.py`.
2. **Training** — `train_cwformer.py`. Verify loss decreases and greedy_cer converges.
3. **Inference** — `inference_cwformer.py` (`CWFormerStreamingDecoder`). Integration test with synthetic audio.
4. **ONNX/Deploy** — `quantize_cwformer.py` + `deploy/inference_onnx.py`. Verify ONNX parity against the PyTorch path.
5. **Benchmarking** — `benchmark_cwformer.py` + `benchmark_random_sweep.py`.

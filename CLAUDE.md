# CWNet-Streaming — Claude Reference Overview

## Project Intent & Goals

CWNet-Streaming is a **causal streaming** neural Morse code (CW) decoder, evolved from the bidirectional CW-Former in [CWNet](https://github.com/parsimo2010/CWNet). It uses a fully causal Conformer architecture (~19.5M params) with CTC loss that processes audio left-to-right with no bidirectional attention.

**Why this exists:** The original CW-Former achieved < 5% CER during training but degraded in real-world streaming due to window stitching artifacts. The sliding-window approach forced a trade-off between accuracy (large windows) and latency (small windows). This project eliminates stitching entirely by using a causal model with KV cache state carry-forward.

**Design philosophy:** Process audio causally (each frame sees only past context), carry state between processing chunks via KV cache and conv buffers, and emit characters as soon as greedy CTC decodes them. No commitment delay is needed — causal CTC guarantees that previously-emitted prefix never changes, so a character can be emitted the moment its frame's argmax is stable.

**Target performance:** 15-40 WPM primary window, any key type (straight, bug, paddle, cootie), SNR > 5-8 dB. Chunk-sized latency (~500 ms typical) from audio to character emission. Desktop CPU/GPU deployment.

**Architecture model:** Causal Conformer, following NVIDIA FastConformer (streaming mode) and Google streaming Conformer. CTC loss with greedy decoding.

**See `PLAN.md`** for the original implementation plan (from the initial bidirectional → causal port).

**See `IMPROVEMENT_PLAN.md`** for the currently-active work: architectural changes (conv kernel 31→63, BatchNorm→LayerNorm in conv module), longer training audio (15s→30s), extended curriculum (WPM 5-50, SNR -5 to 30), auto-curriculum progression, and streaming-correctness bug fixes. Some specifics in this file still describe the pre-refactor state — when in doubt, trust IMPROVEMENT_PLAN.md.

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
  -> Causal ConformerEncoder: 12 blocks (d=256, 4 heads, conv kernel)
     - Self-attention: fully causal (is_causal=True during training, KV cache during inference)
     - Convolution: causal depthwise (left-pad only)
     - Feed-forward: Macaron-style half-step
  -> Linear CTC head -> log_softmax -> incremental greedy decode -> text
```

**Conv kernel size**: currently 31 (620 ms RF at 50 fps). IMPROVEMENT_PLAN
commits to 63 (1240 ms RF) to span a 15 WPM inter-word space in a single
block. After that retrain, 31 → 63.

**Key difference from CWNet:** Attention is fully causal (each frame only attends to past frames), convolutions are left-padded only, and inference uses KV cache + conv buffers for state continuity between chunks. No window stitching. No commitment delay — causal CTC's prefix-stability property makes delayed emission unnecessary, and inference re-decodes the accumulated log_probs each chunk (O(T²) total but correct: past output never changes).

---

## File Map & Key Functions

### Infrastructure (unchanged from CWNet)
- `config.py` — `MorseConfig`, `TrainingConfig`, `create_default_config(scenario)`. **sample_rate = 16000**.
- `vocab.py` — CTC vocabulary (52 classes). `encode(text)`, `decode(indices)`, `decode_ctc(log_probs)`.
- `morse_table.py` — ITU Morse code table + binary trie.
- `morse_generator.py` — Synthetic training data. `generate_sample(config)` -> `(audio_f32, text, metadata)`. All augmentations: AGC, QSB, QRM, QRN, bandpass, HF noise, key types, timing jitter, speed drift.
- `qso_corpus.py` — `QSOCorpusGenerator` for realistic ham radio QSO text.
- `deploy/ctc_decode.py` — Pure-numpy CTC beam search with trigram LM.

### neural_decoder/ — Causal CW-Former

#### Model
- `cwformer.py` — `CWFormer` (~19.5M params): MelFrontend -> Causal ConvSubsampling -> Causal ConformerEncoder -> CTC head.
  - `forward()` — Training: full sequence with causal attention. Input: `(audio, audio_lengths)` -> `(log_probs, output_lengths)`.
  - `forward_streaming(mel_chunk, state)` — Inference: single chunk with KV cache + conv buffers. Returns `(log_probs, new_state)`.
  - `ConvSubsampling` — Causal 2x time reduction (left-pad=2, right-pad=0 in time for both Conv2d layers).
- `conformer.py` — Causal Conformer blocks.
  - `ConformerMHA` — Causal self-attention. Training: `is_causal=True` in SDPA. Inference: KV cache concatenation + causal mask within chunk. RoPE with position offset. RoPE tables auto-extend on demand for long sessions.
  - `ConvolutionModule` — Causal depthwise conv, left-pad only. Inference: conv buffer carry-forward of kernel-1 frames. Uses `nn.BatchNorm1d` today; IMPROVEMENT_PLAN swaps this to `nn.LayerNorm` to eliminate padding-contamination of BN batch stats during training.
  - `ConformerEncoder` — Threads KV caches and conv buffers through 12 blocks. The unused `mask` parameter in `forward()` is a vestige — causality alone handles padding, and the LayerNorm swap in IMPROVEMENT_PLAN removes any remaining reason to propagate it.
- `rope.py` — Rotary Position Embeddings with `offset` parameter for KV cache positions. `_build_tables()` auto-extends the cos/sin table when a streaming session grows longer than the initial `max_len` (4096 default).
- `mel_frontend.py` — `MelFrontend` with `compute_streaming(audio_chunk, stft_buffer)` for incremental mel computation. STFT overlap buffer sized `n_fft - hop_length = 240` samples. Note: `forward()` (training) right-pads audio by `n_fft//2`; `compute_streaming` does not right-pad. IMPROVEMENT_PLAN adds matching right-pad on `flush()` to eliminate the tail asymmetry.

#### Training
- `dataset_audio.py` — `AudioDataset`: streaming IterableDataset (unchanged from CWNet).
- `train_cwformer.py` — Training loop. Micro-batch 8, effective batch 64 via gradient accumulation. Causal attention active during training. Supports optional streaming validation.

#### Inference
- `inference_cwformer.py` — `CWFormerStreamingDecoder`: chunk-based streaming with state carry-forward. No windows, no stitching. No commitment delay — emits characters as soon as greedy CTC decodes them (causal prefix stability). Methods: `feed_audio()`, `get_full_text()`, `flush()`, `decode_file()`, `decode_audio()`. `decode_audio()` peak-normalizes audio to 0.7 on entry to match training distribution; `feed_audio()` does not normalize (caller responsible for live audio gain). `max_cache_sec` parameter caps KV cache to the training-audio max (currently 15 s, becomes 30 s per IMPROVEMENT_PLAN).

### Deployment
- `quantize_cwformer.py` — Streaming ONNX export with state I/O (KV caches + conv buffers as explicit input/output tensors). INT8 dynamic quantization.
- `deploy/inference_onnx.py` — `CWFormerStreamingONNX`: standalone ONNX runtime inference with streaming state management. Supports file, device, and stdin input. **Note**: its `MelComputer.compute_streaming` has known bugs (per-chunk right-pad; fixed-size overlap buffer) that misalign frames at chunk boundaries. IMPROVEMENT_PLAN ports the PyTorch mel logic. Don't use this path for real work until that fix lands.

### Testing
- `tests/test_streaming_equivalence.py` — numerical equivalence check: runs the same audio through `CWFormer.forward()` (training path) and the chunk-by-chunk streaming path, then diffs every intermediate (mel, subsample, each encoder block, log_probs). Also has sub-step diffing inside block 0 (FF1/MHA/conv/FF2/final_norm) for pinpointing divergence. Run after any change to the model or streaming logic.

### Benchmarking
- `benchmark_cwformer.py` — Structured benchmark across SNR, WPM, key types. Phase 1 (clean baseline grid), Phase 2 (single-augmentation marginal effect), Phase 3 (per-character-position error rate for context ramp-up). IMPROVEMENT_PLAN switches Phase 1 to use 100% bandpass (realistic receiver filter) at long audio lengths (min_chars=90, so every sample > 30 s at WPM ≤ 35).
- `benchmark_random_sweep.py` — Random parameter sweep benchmark.

---

## Causal Streaming vs CWNet Bidirectional

| Aspect | CWNet (bidirectional) | CWNet-Streaming (causal) |
|--------|----------------------|--------------------------|
| Attention | Full bidirectional within window | Fully causal (past only) |
| Convolution | Symmetric padding (pad=15 each side) | Left-only padding (pad=30, 0) |
| Inference | Fixed windows + stitching | Chunk-based with KV cache, no stitching |
| Latency | 3-16s (stride + window) | ~chunk size (e.g. 500 ms) + model compute |
| State | Stateless per window | Stateful: KV cache + conv buffers |
| ONNX I/O | mel -> log_probs | mel + state_in -> log_probs + state_out |

---

## Curriculum Learning

The current code reflects the original curriculum (inherited from CWNet).
IMPROVEMENT_PLAN revises these to extend the operating envelope (WPM 5-50
across all stages, SNR down to -5 at full, bandpass probability 0.9→0.6
to teach generalization to both filtered and unfiltered audio, added
random input gain augmentation at ±12 dB for amplitude robustness).

**Current curriculum (pre-IMPROVEMENT_PLAN):**

| Stage | SNR | WPM | AGC | QSB | Key Types | Audio Augmentations |
|-------|-----|-----|-----|-----|-----------|---------------------|
| clean | 15-40 dB | 10-40 | 30% | 0% | 20/20/60/0 S/B/P/C | 10% Farnsworth, 50% bandpass |
| moderate | 8-35 dB | 8-45 | 50% | 25% | 25/25/35/15 S/B/P/C | 20% Farnsworth, 15% QRM, 70% bandpass |
| full | 3-30 dB | 5-50 | 70% | 50% | 30/30/20/20 S/B/P/C | 25% Farnsworth, 30% QRM, 90% bandpass |

See IMPROVEMENT_PLAN.md's "Training curriculum changes" section for the
target values and per-stage random-gain augmentation setup.

---

## Performance Targets
- Primary window (15-40 WPM, any key type, SNR > 8 dB): < 5% CER goal
- Extended (5-50 WPM, low SNR including -5 dB): see
  IMPROVEMENT_PLAN.md success criteria for per-bucket targets.
- Latency: chunk size (e.g. 500 ms) + model compute (~30 ms).
- Real-time factor: < 0.1 (10x faster than real-time on desktop CPU).

---

## Things to Keep in Mind

1. **Sample rate is 16 kHz** — all audio is resampled to 16 kHz internally.
2. **2x subsampling gives 50 fps (20ms per CTC frame)** — resolves dits up to 40+ WPM.
3. **Causal attention** — `is_causal=True` during training, KV cache + causal mask during inference. Never let a frame see future audio.
4. **Causal convolution** — left-pad only. Per-block receptive field = kernel × frame stride. Inference maintains a (kernel-1)-frame conv buffer per layer.
5. **KV cache** — grows per chunk during inference. Trimmed to `max_cache_len` (currently 1500 frames; IMPROVEMENT_PLAN sets this to 1475 so the post-concat attention window stays ≤ training max). Position offset tracks absolute position for RoPE; RoPE tables auto-extend if the session runs long.
6. **No commitment delay** — causal CTC's prefix-stability guarantees the emitted text never changes as more audio arrives, so characters emit as soon as greedy decode stabilizes. Don't reintroduce a delay without a specific reason.
7. **Boundary space tokens** — dataset wraps targets with `[space] + encode(text) + [space]`.
8. **Persistent worker RNG** — use `np.random.default_rng()` (OS entropy), not `worker_info.seed`.
9. **DataLoader tuning** — `persistent_workers=True`, `prefetch_factor=4`. Audio generation is the CPU bottleneck.
10. **Training uses full sequences** — no chunking during training. Inference chunks are for efficiency/latency, not for training. Streaming equivalence to full-forward is verified for in-distribution audio by `tests/test_streaming_equivalence.py`.
11. **Peak normalization** — `morse_generator.generate_sample()` peak-normalizes every training sample to `target_amplitude ∈ [0.5, 0.9]`. `CWFormerStreamingDecoder.decode_audio()` peak-normalizes input to 0.7 to match. The live `feed_audio()` path does NOT normalize (caller is responsible).
12. **Greedy decode only** — no beam search or LM. Beam+LM was tested and didn't help; adds latency without CER gain.
13. **Weights from CWNet (original bidirectional)** used to be shape-compatible — but IMPROVEMENT_PLAN's conv-kernel change (31→63) and BN→LayerNorm swap break that compatibility. New training runs start from scratch.
14. **ONNX state I/O** — 36 state tensors per layer (KV K, KV V, conv buffer) + 2 subsample buffers + position offset. Use per-layer naming: `kv_k_layer0`, `kv_v_layer0`, `conv_buf_layer0`, etc. **Caution**: the ONNX mel frontend has known chunk-boundary bugs; fix per IMPROVEMENT_PLAN before relying on this path.

---

## Implementation Phases

Work through these in order. Each phase should be testable independently:

1. **Core model** — `rope.py` (offset), `conformer.py` (causal attn + conv + state), `cwformer.py` (causal subsampling + `forward_streaming`). Write unit tests for streaming equivalence.
2. **Training** — `train_cwformer.py` updates. Verify training runs and loss decreases.
3. **Inference** — `inference_cwformer.py` (`CWFormerStreamingDecoder`). Integration test with synthetic audio.
4. **ONNX/Deploy** — `quantize_cwformer.py` + `deploy/inference_onnx.py`. Verify ONNX parity.
5. **Benchmarking** — `benchmark_cwformer.py` + `benchmark_random_sweep.py`. Compare CER vs CWNet bidirectional.

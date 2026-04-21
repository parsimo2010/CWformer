# Streaming CW-Former: Causal Streaming Conformer for Morse Decoding

## Key Goals / Overarching Mission

**The problem:** The current bidirectional CW-Former (~19.5M params, Conformer + CTC) achieves excellent accuracy during training (< 5% CER) but degrades in real-world streaming because it relies on decoding fixed windows of audio and stitching the results together. This forces a painful trade-off: short windows/stride = poor accuracy from stitching errors, long windows/stride = unacceptable latency (seconds of delay before new characters appear). Neither option is good for real-time use.

**The goal:** Build a new model that processes audio **causally** (left-to-right, no bidirectional attention) with **state carry-forward** between chunks, eliminating window stitching entirely. The model should:

1. **Decode in real-time** as audio streams in, using only past context (fully causal attention and convolution). No frame ever sees future audio.
2. **Emit characters as soon as greedy CTC decodes them.** The original plan called for a commitment delay (wait for inter-character space before emitting), but that turns out to be unnecessary: because the model is fully causal, greedy CTC decode of a prefix is always a correct prefix of the full decode. Past output never changes as more audio arrives. The implementation just re-decodes the accumulated log-probs on every chunk; characters emit the moment their frame's argmax stabilizes (which the model learns via CTC blanks). See `neural_decoder/inference_cwformer.py` for the current behavior.
3. **Achieve comparable accuracy** to the current bidirectional model. Expected: 1-3% CER increase from the causal constraint at moderate SNR, offset by eliminating stitching artifacts (net real-world accuracy should improve).
4. **Keep total latency low** — chunk size + model processing. With a 500 ms chunk and ~30 ms CPU compute, end-to-end latency is about 530 ms. No commitment delay is added.
5. **Preserve the training infrastructure** that works well: synthetic data generation with all augmentations, CTC loss, AMP, SpecAugment, cosine LR with warmup, curriculum learning (clean -> moderate -> full), gradient accumulation, data buffering.
6. **Preserve the deployment pipeline**: ONNX export (INT8 quantized), pure-numpy inference without PyTorch, live audio device/stdin/file input.
7. **Enable fine-tuning** from the existing bidirectional checkpoint (weights are shape-compatible) rather than requiring training from scratch.

**The approach:** A fully causal Conformer (same ~19.5M param architecture: d=256, 12 layers, 4 heads, conv kernel=31) with KV cache for attention state and conv buffers for continuity across processing chunks. This will be implemented in a new repository, copying unchanged infrastructure files from CWNet.

---

## Context

**Architecture model:** Causal Conformer, following the approach used in NVIDIA's FastConformer (streaming mode) and Google's streaming Conformer. This is the simplest, most proven streaming architecture — it's the standard Conformer with causal self-attention and causal convolutions. The Conformer's dual mechanism (self-attention for long-range patterns + convolution for local temporal patterns like dits/dahs/gaps) is ideal for Morse.

**Why fully causal works for Morse:** Unlike speech (where coarticulation makes future context critical), Morse characters are naturally delimited by silence gaps. Human operators decode CW left-to-right. The model will learn to emit CTC blanks during a character and commit when it sees the inter-character gap — the same strategy used by human decoders. The conv module's 620ms receptive field captures complete character patterns locally.

**No commitment delay:** An earlier draft of this plan called for an ICS-confirmation delay of up to ~2 s before emitting each character, to avoid committing to a shared-prefix character (E/I/S/H/5 all start with a dit) before enough context had arrived. The final implementation does not use one. For a fully causal model with greedy CTC, past output is a function of past audio alone — once a frame's argmax is the first non-blank token of a character, nothing the model sees later can change the already-emitted prefix. The decoder just re-decodes the accumulated log-probs each chunk and diffs against what was already emitted; the model's own CTC blanks learn to stall on ambiguous prefixes (`.` remains blank-heavy until the model sees whether a second dit follows).

**Expected accuracy impact:** 1-3% CER increase at moderate SNR (10-30 dB), 3-5% at low SNR (5-10 dB) vs fully bidirectional. Net real-world accuracy should still **improve** because stitching artifacts (the current biggest problem) are eliminated entirely.

**Latency:** Configurable via chunk size alone. A 500 ms chunk yields ~530 ms end-to-end latency on desktop CPU (audio accumulation + model forward). No commitment delay is added.

---

## Architecture

```
Audio (16 kHz mono, streaming)
  -> MelFrontend: incremental log-mel (40 bins, 200-1400 Hz, 25ms/10ms) + SpecAugment (train only)
  -> Causal ConvSubsampling: 2x time reduction with boundary buffers
  -> Causal ConformerEncoder: 12 blocks (d=256, 4 heads)
       - Self-attention: FULLY CAUSAL (each frame sees only past frames + itself)
         - Training: is_causal=True in SDPA (triggers causal Flash Attention kernel)
         - Inference: KV cache for past chunks + causal mask within current chunk
       - Convolution: causal depthwise conv (kernel=31, left-pad=30, right-pad=0)
         - 620ms of past local context per frame
       - Feed-forward: unchanged (pointwise, no temporal dependency)
  -> Linear CTC head -> log_softmax -> incremental greedy decode -> text
```

**Parameters:** ~19.5M (identical to current model — same dimensions, same layer counts). Weights from existing bidirectional checkpoints load directly (tensor shapes match exactly; the only difference is runtime masking/padding behavior).

**Why NOT an RNN/GRU/LSTM:** The user's prior experience with a causal GRU/LSTM (few million params) showed inferior accuracy. This is due to fundamental architecture limitations: RNNs compress all history into a fixed-size hidden state, losing information over time. The Conformer's self-attention directly accesses any past frame through the KV cache — no information compression. Even at equal parameter counts, Conformers outperform RNNs on sequence-to-sequence tasks. Staying with the Conformer architecture preserves the accuracy gains already achieved.

**Why NOT encoder-decoder (Whisper-style):** Whisper uses 30s windows with an autoregressive decoder — it has the same stitching problem. Its encoder-decoder architecture adds complexity (prediction network, cross-attention, autoregressive decoding loop) without benefit for Morse, where the vocabulary is small (52 tokens) and CTC provides natural time-aligned output. CTC is simpler, faster, and sufficient for Morse's clear character boundaries.

---

## Detailed Changes by File

### Files to copy unchanged
- `vocab.py` — 52-class vocabulary, encode/decode, greedy CTC decode
- `morse_table.py` — ITU Morse code tables
- `morse_generator.py` — Synthetic audio generation with all augmentations
- `config.py` — MorseConfig, TrainingConfig, scenario presets
- `qso_corpus.py` — QSO corpus text generator

### Files to copy and modify

#### 1. `neural_decoder/rope.py` — Add position offset for KV cache

**Change:** Add `offset: int = 0` parameter to `apply_rope()` and `RotaryEmbedding.forward()`.

When using KV cache during inference, new Q/K vectors must get RoPE at their actual positions `[offset, offset+1, ..., offset+chunk_len-1]` rather than `[0, 1, ...]`. Cached K vectors already have correct positions baked in from prior RoPE application — they don't need re-rotation.

```python
# In apply_rope():  change cos/sin slicing
#   Before: cos = cos[:seq_len]
#   After:  cos = cos[offset:offset+seq_len]

# In RotaryEmbedding.forward():  add offset, pass through
def forward(self, q, k, offset=0):
    return apply_rope(q, self.cos, self.sin, offset), \
           apply_rope(k, self.cos, self.sin, offset)
```

#### 2. `neural_decoder/conformer.py` — Causal attention, causal conv, state I/O

This is the largest change. Three sub-modules need modification:

**2a. `ConformerMHA` — Fully causal self-attention with KV cache**

*Training:* Use `is_causal=True` in `F.scaled_dot_product_attention()`. This triggers PyTorch's built-in causal Flash Attention kernel — the most efficient path. No explicit mask tensor construction needed. Each frame attends only to itself and all preceding frames.

```python
# Training forward (kv_cache is None):
out = F.scaled_dot_product_attention(
    q, k, v, dropout_p=dropout_p, is_causal=True
)
```

*Inference:* Accept optional `kv_cache = (k_cached, v_cached)` each of shape `(B, H, T_cached, d_k)`.

```python
# Inference forward (kv_cache provided):
# 1. Compute Q, K, V for new chunk only (C frames)
# 2. Apply RoPE with offset = T_cached
# 3. Concatenate: K_full = cat([k_cached, k_new], dim=2)
#                 V_full = cat([v_cached, v_new], dim=2)
# 4. Build causal mask for current chunk attending to full KV:
#    - All cached positions: always attend (they're all in the past)
#    - Current chunk positions: causal (frame i sees only frames 0..i within chunk)
#    mask shape: (C, T_cached + C)
#    past_part = zeros(C, T_cached)     # all attend
#    chunk_part = triu(-inf, diagonal=1) of (C, C)  # causal within chunk
#    mask = cat([past_part, chunk_part], dim=1)
# 5. SDPA(Q_new, K_full, V_full, attn_mask=mask)
# 6. Return output + (K_full, V_full) as new cache
```

Updated forward signature:
```python
def forward(self, x, mask=None, kv_cache=None, pos_offset=0):
    # Returns: (output, new_kv_cache_or_None)
```

When `kv_cache is None` (training or full-sequence eval): use `is_causal=True`.
When `kv_cache is not None` (streaming inference): use constructed mask + cache concatenation.

*KV cache memory per layer:* `2 * B * H * T_cached * d_k * 4 bytes`. With T_cached=1500 (30s), B=1, H=4, d_k=64: **3 MB/layer, 36 MB total** for 12 layers. Acceptable.

**2b. `ConvolutionModule` — Causal depthwise convolution with buffer**

Change depthwise conv from symmetric padding to causal (left-only):

```python
# Before: self.depthwise = Conv1d(..., padding=kernel//2)  # symmetric, sees future
# After:  self.depthwise = Conv1d(..., padding=0)           # no built-in padding
#         In forward: x = F.pad(x, (kernel-1, 0))           # left-pad only
```

This means each frame's conv output depends only on itself and the 30 preceding frames (620ms of past context). Sufficient to detect dits, dahs, and inter-character gaps at all target speeds.

*Inference:* Accept `conv_buffer` of shape `(B, D, kernel-1)` = `(B, 256, 30)`. Prepend buffer to input before depthwise conv. Save last 30 frames of pre-conv input as new buffer.

```python
def forward(self, x, conv_buffer=None):
    # ... after GLU, before depthwise:
    if conv_buffer is not None:
        depthwise_input = torch.cat([conv_buffer, x_after_glu], dim=2)
        new_buffer = x_after_glu[..., -30:]
    else:
        depthwise_input = F.pad(x_after_glu, (30, 0))  # training: left-pad with zeros
        new_buffer = None
    out = self.depthwise(depthwise_input)
    # Returns: (output, new_buffer)
```

*Buffer memory per layer:* `256 * 30 * 4 = 30 KB/layer`, **360 KB total**. Negligible.

**Note on BatchNorm:** BatchNorm1d uses running statistics in eval mode. Works correctly per-chunk with no changes. During training, batch statistics computed over full sequences — also fine.

**2c. `ConformerBlock` and `ConformerEncoder` — Thread state through**

Updated signatures to pass and collect state:

```python
# ConformerBlock:
def forward(self, x, mask=None, kv_cache=None, conv_buffer=None, pos_offset=0):
    # Returns: (x, new_kv_cache, new_conv_buffer)

# ConformerEncoder:
def forward(self, x, mask=None, kv_caches=None, conv_buffers=None, pos_offset=0):
    # kv_caches: list of 12 (k, v) tuples, or None (training)
    # conv_buffers: list of 12 tensors, or None (training)
    # Returns: (x, new_kv_caches, new_conv_buffers)
```

When `kv_caches` and `conv_buffers` are None (training), the forward pass works exactly as before except with `is_causal=True` and left-only conv padding. No state returned.

**2d. `ConformerConfig` — Add streaming fields**

```python
@dataclass
class ConformerConfig:
    # ... existing fields ...
    max_cache_len: int = 1500    # Max KV cache frames (~30s at 50fps)
```

#### 3. `neural_decoder/cwformer.py` — Causal ConvSubsampling + streaming forward

**3a. `ConvSubsampling` — Causal padding**

Change both Conv2d layers from symmetric padding to causal (left-only in time, symmetric in frequency):

```python
# Both convs: change from padding=1 to padding=0
# Before each conv, apply:  F.pad(x, (1, 1, 2, 0))
#   (freq_left=1, freq_right=1, time_left=2, time_right=0)
# With kernel=3: output frame t sees input frames [t-2, t-1, t] in time (causal)
# Frequency dimension keeps symmetric padding (no temporal concern)
```

*Inference buffers:* Conv1 needs last 2 mel frames. Conv2 needs last 2 post-conv1 frames.
Sizes: `(B, 1, 2, 40)` and `(B, 256, 2, 20)`.

**3b. `CWFormer` — Add `forward_streaming()` method**

Keep existing `forward()` for training. The only training change: the encoder now internally uses `is_causal=True` and causal conv padding (no explicit mask construction needed in the model). Add new streaming method:

```python
def forward_streaming(self, mel_chunk, state):
    """Process one chunk of mel frames through the causal model.
    
    Args:
        mel_chunk: (B, T_chunk, n_mels) new mel frames
        state: dict with:
            'kv_caches': list of 12 (k, v) tuples or None (first chunk)
            'conv_buffers': list of 12 conv buffer tensors or None
            'subsample_buffers': tuple of conv subsampling boundary buffers
            'pos_offset': int, cumulative CTC frame position
    
    Returns:
        log_probs: (T_out, B, C) CTC log-probs for this chunk's frames
        new_state: dict with updated state
    """
```

**3c. `CWFormerConfig` — Add streaming config**

```python
@dataclass
class CWFormerConfig:
    # ... existing fields ...
    inference_chunk_ms: int = 1000  # Default chunk size for inference (ms)
```

#### 4. `neural_decoder/mel_frontend.py` — Streaming mel computation

Add `compute_streaming()` method to `MelFrontend`:

```python
def compute_streaming(self, audio_chunk, stft_buffer=None):
    """Compute mel for new audio chunk, maintaining STFT overlap state.
    
    Args:
        audio_chunk: (B, N_new) new audio samples
        stft_buffer: (B, n_fft - hop_length) = (B, 240) overlap from previous chunk
    
    Returns:
        mel: (B, T_new, n_mels) new mel frames
        new_buffer: (B, 240) saved for next call
    """
```

The STFT (window=400, hop=160) needs 240 samples of overlap between chunks to form complete windows at boundaries. SpecAugment is NOT applied during streaming inference (already gated by `self.training`).

#### 5. `neural_decoder/train_cwformer.py` — Minimal changes

1. **Causal attention during training:** No explicit mask needed. The model's `ConformerMHA` uses `is_causal=True` in SDPA during training. The conv module uses causal left-padding. These are internal to the model — the training loop doesn't change.

2. **Streaming validation (optional):** Every N epochs, run a few validation samples through `forward_streaming()` chunk-by-chunk and compare decoded text with single-pass `forward()`. This catches state management bugs early. Differences indicate implementation errors (should match within floating-point tolerance).

3. **Everything else preserved:** AMP, cosine LR with warmup, gradient accumulation (micro-batch 8, effective 64), SpecAugment, curriculum stages (clean/moderate/full), buffer/cache reuse system, checkpointing, 20K samples/epoch — all unchanged.

4. **Fine-tuning from existing checkpoint:** Weights load directly via `model.load_state_dict()` (all tensor shapes identical). The model adapts to the causal constraint during fine-tuning. Recommend starting from best bidirectional checkpoint, 50-100 epochs on "full" scenario.

#### 6. `neural_decoder/inference_cwformer.py` — Major rewrite (streaming, no stitching)

Replace `CWFormerDecoder` (window+stitch) with `CWFormerStreamingDecoder`:

```python
class CWFormerStreamingDecoder:
    """Streaming causal decoder for CW-Former.
    
    Processes audio chunk-by-chunk with state carry-forward.
    No windowing. No stitching. Characters emitted incrementally.
    """
    
    def __init__(self, checkpoint, chunk_ms=1000, device='cpu', max_cache_sec=30.0):
        # Load model in eval mode, initialize empty state
        
    def reset(self):
        """Reset all state for a new decoding session."""
        
    def feed_audio(self, audio_chunk: np.ndarray) -> str:
        """Feed raw audio samples, return NEW characters decoded since last call.
        
        Accumulates audio until a full chunk is ready, then processes.
        Returns empty string if chunk not yet complete.
        """
        
    def get_full_text(self) -> str:
        """Get all decoded text so far."""
        
    def flush(self) -> str:
        """Process any remaining buffered audio (partial chunk). Call at end of stream."""
        
    def decode_file(self, path: str) -> str:
        """Decode complete audio file by feeding as streaming chunks."""
        
    def decode_audio(self, audio: np.ndarray) -> str:
        """Decode complete audio array by feeding as streaming chunks."""
```

**Processing pipeline per chunk:**
1. Accumulate raw audio in internal buffer until chunk is ready (default 1s = 16,000 samples)
2. Compute mel frames for the chunk with STFT overlap buffer
3. Run mel through causal ConvSubsampling with boundary buffers -> ~25 CTC frames per 500ms
4. Run through Conformer encoder with KV cache + conv buffers (causal attention)
5. Apply CTC head -> log_probs for this chunk's frames
6. Append new log_probs to running CTC output buffer
7. Run greedy decode on recent CTC buffer to detect newly committed characters
8. Emit new characters; update all state

**Character emission logic (no commitment delay):**

The implementation emits characters as soon as greedy CTC decode produces them — there is no explicit commitment horizon. The reasoning:

- The model is fully causal, so a CTC frame at time `t` is a deterministic function of audio `[0, t]`. More audio arriving at time `> t` cannot change the argmax at frame `t`.
- Greedy CTC decode of a prefix is therefore always a correct prefix of the full decode. If the decoder emits "E" at chunk boundary `k`, that "E" stays in the transcript regardless of what follows.
- The model learns to stall on ambiguous prefixes on its own: after a single dit, the CTC output remains blank-heavy until the model sees whether a second dit follows. "E" only becomes the argmax after the inter-character space has arrived in the frame's left-context — i.e., the model implements ICS-confirmation in its weights, not in an external delay loop.

Implementation:
- Maintain a running CTC log_prob buffer.
- After each processing chunk, run greedy CTC decode on the full accumulated buffer.
- Diff against what was already emitted — new suffix is the "newly decoded characters" return value.
- This is `O(T²)` total decode work across a session, but each decode is cheap (greedy argmax over ~50 fps × num_classes).

Latency = chunk accumulation + model compute only.

**KV cache trimming:** When `pos_offset` exceeds `max_cache_len` (default 1500 = 30s), slice oldest frames from all KV caches: `k_cache = k_cache[:, :, -max_cache_len:, :]`. Position counter continues incrementing — RoPE handles this correctly since cached K vectors already have their absolute positions encoded.

#### 7. `quantize_cwformer.py` — ONNX export with state I/O

Create `_CWFormerStreamingCore` module for ONNX export.

**ONNX Inputs:**
- `mel_chunk`: `(1, T_chunk, 40)` — float32, new mel frames
- `pos_offset`: `(1,)` — int64, current position in stream
- Per-layer state (12 layers x 3 tensors = 36 state tensors):
  - `kv_k_layer{i}`: `(1, 4, T_cached, 64)` — float32
  - `kv_v_layer{i}`: `(1, 4, T_cached, 64)` — float32
  - `conv_buf_layer{i}`: `(1, 256, 30)` — float32
- Subsample buffers (2 tensors):
  - `sub_buf1`: `(1, 1, 2, 40)` — float32 (conv1 time boundary)
  - `sub_buf2`: `(1, 256, 2, 20)` — float32 (conv2 time boundary)

**ONNX Outputs:**
- `log_probs`: `(T_out, 1, 52)` — float32, CTC log-probs for this chunk
- All state tensors (updated versions of inputs)
- `pos_offset_out`: `(1,)` — int64

**Dynamic axes:** `T_chunk` and `T_cached` are dynamic. Batch fixed at 1 for deployment.

Export process: FP32 ONNX (opset 17) -> verify vs PyTorch -> INT8 dynamic quantization -> benchmark.

Note: The causal attention mask within the current chunk can be constructed inside the ONNX graph using standard ops (triu, where), or precomputed for a fixed chunk size.

#### 8. `deploy/inference_onnx.py` — Streaming ONNX inference

Replace `CWFormerONNX` with `CWFormerStreamingONNX`:

```python
class CWFormerStreamingONNX:
    """Streaming CW decoder using ONNX Runtime. No PyTorch required."""
    
    def __init__(self, model_path, config_path=None, chunk_ms=1000, max_cache_sec=30.0):
        self.mel = MelComputer(config)  # pure numpy (existing, unchanged)
        self.session = ort.InferenceSession(model_path)
        self._state = self._init_state()
        
    def _init_state(self):
        """Create zero-initialized state tensors (empty KV cache, zero conv buffers)."""
        
    def feed_audio(self, audio_chunk):
        """Feed audio chunk, return new decoded characters."""
        
    def decode_live(self, audio_source, display=None):
        """Stream from device/stdin/file with live display."""
```

Live streaming becomes dramatically simpler: feed audio chunks into `feed_audio()`, display returned characters. No window tracking, no stitching heuristics, no overlap management.

#### 9. Benchmark files — Add streaming mode

Add `--streaming` flag to `benchmark_cwformer.py` and `benchmark_random_sweep.py`. When set, use `CWFormerStreamingDecoder` instead of the old window+stitch decoder. Add streaming-specific metrics:
- Per-chunk processing latency (ms)
- Real-time factor (processing time / audio duration)
- Peak memory usage (RSS)
- CER comparison: causal streaming vs bidirectional (quantify accuracy trade-off)

#### 10. `dataset_audio.py` — No changes

Training sample generation is unchanged. Streaming is purely an inference concern. The training loop processes full sequences with causal attention mask applied internally by the model.

---

## Training Strategy

### Phase 1: Fine-tune from existing checkpoint (recommended)
1. Load best bidirectional CW-Former checkpoint into the causal model
2. `model.load_state_dict(checkpoint['model'])` — all tensor shapes identical, loads cleanly
3. Fine-tune 50-100 epochs on "full" scenario with causal attention
4. The model adapts to predict using only past context
5. Monitor CER convergence; expect rapid adaptation (bidirectional weights are a strong starting point; the model mainly needs to learn to "wait for the gap" before committing)
6. Use the existing curriculum's best checkpoint as starting point

### Phase 2: Train from scratch (if fine-tuning insufficient or for ablation)
1. Clean -> moderate -> full curriculum exactly as before
2. Causal constraint active from the start (model never sees future frames)
3. Same hyperparameters: AdamW (lr=3e-4, wd=0.01, betas=0.9/0.98), cosine LR with warmup, AMP
4. May benefit from slightly more training epochs since the model has a harder task

### Curriculum stages: unchanged
| Stage | Epochs | SNR | WPM | Augmentations |
|-------|--------|-----|-----|---------------|
| clean | 200 | 15-40 dB | 10-40 | Mild (AGC 30%, farnsworth 10%) |
| moderate | 300 | 8-35 dB | 8-45 | Moderate (QSB 25%, QRM 15%, bandpass 70%) |
| full | 500 | 3-30 dB | 5-50 | Heavy (all augmentations 25-50%) |

---

## Verification Plan

### Unit tests (run during development)
1. **Streaming equivalence:** Process a full 10s sequence through `model.forward()` (causal, single pass). Process the same mel spectrogram chunk-by-chunk through `model.forward_streaming()` with state carry. Verify CTC log_probs match within floating-point tolerance (< 1e-5 absolute diff). This is the critical correctness test.
2. **Causal conv buffer:** Full-sequence causal conv vs chunk-by-chunk with buffer. Must match exactly (deterministic integer indexing).
3. **RoPE offset:** Verify `apply_rope(x, cos, sin, offset=N)` matches `apply_rope(full_x, cos, sin)[..., N:N+len, :]` for the corresponding positions.
4. **KV cache trimming:** Process 60s of audio, trim cache at 30s boundary. Verify output for frames near (but after) the trim point is close to untrimmed output (< 1% relative error for recent frames; older frames may diverge).

### Integration tests
1. **Deterministic roundtrip:** Generate a clean CW sample (30 dB SNR, 20 WPM, known text) -> stream through decoder -> verify decoded text matches reference.
2. **Chunk-size invariance:** Decode the same audio file with chunk sizes of 250ms, 500ms, 1s, 2s. All should produce identical decoded text (up to edge effects at the very end from partial final chunks).
3. **ONNX parity:** PyTorch streaming decoder vs ONNX streaming decoder on the same audio -> identical CTC log_probs (< 0.01 max diff) and identical decoded text.

### Accuracy benchmarks
1. **Causal vs bidirectional:** Full benchmark grid (SNR x WPM x key type) comparing new causal model vs old bidirectional model. Target: < 3% CER increase at SNR > 10 dB, < 5% at SNR 5-10 dB.
2. **Streaming vs old window+stitch:** Compare the new streaming decoder vs old CWFormerDecoder (window+stitch) on the same test set. This measures the net real-world impact (causal penalty offset by elimination of stitching errors).

### Performance benchmarks
1. **Per-chunk latency:** Time to process one 1s chunk on desktop CPU. Target: < 50ms (20x faster than real-time).
2. **Memory:** Peak RSS with 30s KV cache. Target: < 200 MB.
3. **Real-time factor:** Process 100x 30s audio files, measure total RTF. Target: < 0.1.

---

## New Repository File List

```
CWformer/
  config.py                          # COPY unchanged
  vocab.py                           # COPY unchanged
  morse_table.py                     # COPY unchanged
  morse_generator.py                 # COPY unchanged
  qso_corpus.py                      # COPY unchanged
  neural_decoder/
    mel_frontend.py                  # COPY + add compute_streaming() method
    rope.py                          # COPY + add offset parameter to apply_rope/forward
    conformer.py                     # MODIFY: is_causal=True, causal conv, KV cache, state I/O
    cwformer.py                      # MODIFY: causal ConvSubsampling, forward_streaming()
    dataset_audio.py                 # COPY unchanged
    train_cwformer.py                # COPY + add streaming validation, minimal changes
    inference_cwformer.py            # REWRITE: CWFormerStreamingDecoder (no stitching)
  quantize_cwformer.py               # REWRITE: streaming ONNX with state I/O
  deploy/
    inference_onnx.py                # REWRITE: CWFormerStreamingONNX
  benchmark_cwformer.py              # COPY + add --streaming flag
  benchmark_random_sweep.py          # COPY + add --streaming flag
```

---

## Summary of Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Architecture | Causal Conformer (same base as current) | Proven accuracy for CW; attention >> RNN; simplest streaming architecture |
| Attention | Fully causal (is_causal=True) + KV cache | Each frame sees only past; matches how humans decode CW left-to-right |
| Convolution | Causal (left-pad=30, right-pad=0) | 620ms past receptive field; detects dits/dahs/gaps without future context |
| Loss function | CTC (unchanged) | Simple, sufficient for Morse; natural frame-to-time alignment |
| Chunk size | Default 1s (~50 CTC frames), configurable | Good latency/efficiency balance; 200ms-2s range supported |
| KV cache limit | 30 seconds (1500 frames) | Caps memory at ~36 MB; more than enough Morse context |
| Decoding | Incremental greedy CTC (no stitching) | Eliminates stitching entirely; state carry-forward = no window boundaries |
| Model size | ~19.5M params (unchanged) | Same dimensions; existing weights transfer directly |
| Training | Fine-tune from bidirectional checkpoint | Weights compatible (same shapes); rapid convergence expected |
| Comparison model | NVIDIA FastConformer / Google streaming Conformer | Well-established causal streaming architecture for sequence-to-sequence |

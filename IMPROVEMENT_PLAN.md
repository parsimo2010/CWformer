# CWformer Improvement Plan

This plan is written to be executed in a fresh Claude Code session. Read
[CLAUDE.md](CLAUDE.md) first, then this file. The plan is self-contained —
don't rely on any prior conversation to fill in gaps.

---

## Additional bugs found in code audit (incorporated below)

A separate code audit surfaced several latent issues. The ones worth
fixing (in priority order) are now woven into the plan below:

- **BN pollution from unmasked padding** (*biggest training-time bug*):
  `ConvolutionModule` uses `nn.BatchNorm1d`, which reduces over `(B, T)`
  per channel. The training `forward()` builds a length mask but throws
  it away — padded positions flow through attention, conv, and BN. BN's
  EMA running stats inherit the contamination, and variable-length
  batches mean this is not small. **Fix: swap BN → LayerNorm in the
  conv module.** Details in Architecture section.
- **ONNX streaming mel is broken at chunk boundaries**: right-pads
  every chunk with `n_fft//2` zeros (training only pads right at the
  end), and uses a fixed 240-sample overlap buffer instead of the
  `audio[consumed_up_to:]` the PyTorch path uses. Every chunk boundary
  misaligns. Details in Streaming correctness section.
- **Mel tail asymmetry**: `forward()` pads both sides with `n_fft//2`;
  `compute_streaming()` left-pads only, and the final flush doesn't
  right-pad. Causes 1–2 missing frames at the tail and hides tail bugs
  from the equivalence test.
- **KV-cache lets relative position briefly exceed training range**:
  after concat, window is `T_cached + T_chunk` (e.g. 750 + 25 = 775),
  but training only saw up to 749. Tiny RoPE extrapolation every chunk.
  Fix by trimming to `max_cache_len - chunk_size` (or trimming after
  concat).
- **Doc drift**: CLAUDE.md and PLAN.md describe a ~2 s commitment delay
  that isn't in the code. The code correctly re-decodes accumulated
  log-probs each chunk (causal CTC → past output never changes).
  Update the docs.
- **Vocab docstring wrong**: indices listed as 38–55 / 56–63 but
  actually 38–45 / 46–51. Cosmetic but misleading.
- **Inference-time amplitude normalization** (already fixed this
  session): `decode_audio` now peak-normalizes to 0.7. For
  robustness against arbitrary real-world gain, also add random-gain
  augmentation at training input.

**Not changing** (checked and confirmed correct):
- `is_causal=True` (training) vs explicit causal mask + KV-cache
  (streaming) is mathematically equivalent at valid positions.
- `ConvSubsampling.forward_streaming` carry logic (parity-aware
  carry=1 or 2) is correct — traced multiple chunk sequences and
  outputs align 1:1 with full-forward at the same absolute positions.
- Conv-module 30-frame buffer preserves the `F.pad(out, (30, 0))`
  equivalence.
- `_process_chunk` re-decoding is O(T²) but correct — causal CTC
  log-probs are frozen once written.

---

## Context: what we learned and why these changes matter

The current checkpoint was trained on audio capped at 15 s with heavy
augmentation (`bandpass_probability=0.9`, `agc_probability=0.7`,
`hf_noise_probability=0.5`, etc.) and shows:

- **Training val CER ≈ 7%** on augmented synthetic audio.
- **Benchmark Phase 1 CER ≈ 22%** on *clean* synthetic audio in the same
  WPM/SNR range. The model is adapted to augmented audio and degrades on
  clean audio (confirmed via Phase 2: adding bandpass at test time drops
  CER from 13% to 5.5%).
- **Streaming vs full-forward numerical equivalence**: confirmed correct
  for audio ≤ 15 s. For longer audio the training-path reference itself
  does OOD attention (RoPE positions beyond what the model saw). Streaming
  is not broken; the model was trained on too-short audio to generalize.
- **Low-WPM weakness**: WPM=15 cells in benchmark have 28–80% CER even at
  SNR=30. The conv receptive field (620 ms at kernel=31, 50 fps) does not
  span an inter-word space at 15 WPM (840 ms) or 10 WPM (1680 ms). The
  model has to rely entirely on attention to span gaps at low WPM, which
  it handles poorly given 15 WPM samples in training are only ~17 chars
  (so there's not much low-WPM data per sample).

**Real-world target**: the decoder runs on live CW audio — callsigns,
signal reports, conversational exchanges. Minimum-useful capture is a
short CQ or signal report (~10–15 chars ≈ 5–20 s depending on WPM).
Medium-useful is a full callsign exchange (~30–40 chars ≈ 15–60 s). Rare
but plausible: a contest exchange or quick QSO (up to ~90 chars). The
model must handle this spread well, not just the short end.

---

## Goals (in priority order)

1. **Fix the training/inference length mismatch** so that typical
   real-world CW captures are in-distribution.
2. **Fix the low-WPM architectural weakness** by giving the conv module
   enough receptive field to span a 10 WPM inter-word space.
3. **Fix the clean-audio OOD gap** by reducing augmentation dominance
   so the model generalizes to both filtered and unfiltered audio.
4. **Make training hands-off** via automated curriculum progression, so
   overnight/multi-day training doesn't need babysitting.
5. **Make the benchmark honest** — test at realistic audio lengths with
   training-consistent filtering, so the numbers reflect deployment
   performance, not OOD weirdness.

---

## Architecture changes

### Conv kernel: 31 → 63

**Why 63 specifically**: at 50 fps (20 ms per subsampled frame), kernel=63
gives 62 frames of past context = **1240 ms receptive field** per
Conformer block. This comfortably spans:

- 10 WPM inter-word space (1680 ms → covered by ~1.35 blocks stacking)
- 15 WPM inter-word space (840 ms → fully within one block)
- 40 WPM full character like "0" (5 dahs + 4 spacings at 30 ms unit =
  570 ms → fully within one block)

5 WPM IWS (1680 ms) still needs ~2 stacked blocks to span, but that's
fine — stacked conv RF is (kernel-1)·N_layers+1, which at 12 layers and
kernel=63 gives 7.44 s. More than enough.

**Kernel must be odd** — see the assert in `ConvolutionModule.__init__`.

**Retraining required**: the stored conv weights are shape-incompatible
with kernel=63. Start training from scratch for the `clean` scenario.
Don't try to port the old weights; the learning dynamics will be
different with a wider kernel and the model needs to relearn from the
new curriculum anyway.

**Files to change**:
- [config.py](config.py): default is set in `MorseConfig` but kernel is a
  Conformer config field. The change lives in
  [neural_decoder/conformer.py:51](neural_decoder/conformer.py#L51)
  (`conv_kernel: int = 31` → `63`) and the CLI default in
  [neural_decoder/train_cwformer.py:718](neural_decoder/train_cwformer.py#L718)
  (`--conv-kernel`, default 31 → 63).

### BatchNorm1d → LayerNorm in ConvolutionModule

**Why**: the current `ConvolutionModule` uses `nn.BatchNorm1d` after the
depthwise conv
([neural_decoder/conformer.py:246](neural_decoder/conformer.py#L246)).
BN reduces over `(B, T)` per channel. The training `forward()` builds a
length mask at
[neural_decoder/cwformer.py:268-276](neural_decoder/cwformer.py#L268-L276)
then never applies it — padded positions run through attention, conv,
and BN, contaminating BN's batch statistics every training step and
polluting the EMA running stats used at eval. With micro-batch 8 and
audio lengths up to 30 s (with variable shorter samples), the padding
fraction is large.

Google's streaming Conformer uses LayerNorm in the conv module for
exactly this reason — LayerNorm is per-frame, so padded frames don't
affect valid frames' normalization statistics.

**Change**: replace `self.batch_norm = nn.BatchNorm1d(d_model)` with
`self.layer_norm_conv = nn.LayerNorm(d_model)`. The LayerNorm expects
input shape `(B, T, D)`, so the transpose dance needs one extra swap:

```python
# After depthwise conv, before activation:
out = out.transpose(1, 2)       # (B, T, D)
out = self.layer_norm_conv(out) # (B, T, D)
out = out.transpose(1, 2)       # (B, D, T)  ← back for pointwise2
out = F.silu(out)               # Swish as before
```

Or restructure to work in `(B, T, D)` throughout the conv module. Either
is fine; keep it readable.

**Retraining impact**: this is a weight-shape change in that
`BatchNorm1d` has `(D,)` running mean/var + `(D,)` scale/bias, while
`LayerNorm(D)` has `(D,)` scale/bias and no running stats. So the old
checkpoint's BN parameters won't load. Since we're already retraining
from scratch for the conv kernel change, this is a free upgrade.

**Side benefit**: eliminates the train/eval mode mismatch that BN
introduces. Streaming inference with `model.eval()` will behave more
consistently.

**Files to change**:
- [neural_decoder/conformer.py](neural_decoder/conformer.py): the
  `ConvolutionModule` — replace BN with LayerNorm.

### max_audio_sec: 15 → 30

**Why 30 s**: covers short QSOs (~40 chars at 15 WPM ≈ 32 s) and typical
contest exchanges. Going longer (60+ s) doubles training attention memory
(it's O(T²)) and runs into diminishing returns — the vast majority of
real-world captures that start cleanly are under 30 s. The sweet spot is
generous enough to give the model long-sequence training but tight enough
that a single A100/4090 can train it at reasonable batch size.

**Memory implication**: attention is O(T²) per head per layer. Going from
15 s (750 frames) to 30 s (1500 frames) is a 4× memory increase on the
attention computation. Training-time mitigation:

- `batch_size` stays at 8 (micro-batch). If OOM, drop to 4 and bump
  `--accum-steps` to 16 to keep effective batch = 64.
- AMP (bf16/fp16 autocast) is already on; keep it.
- `max_cache_len` in ConformerConfig: default is 1500, already sized for
  30 s audio. No change needed, but verify
  [neural_decoder/conformer.py:54](neural_decoder/conformer.py#L54).

**Files to change**:
- [neural_decoder/train_cwformer.py:728](neural_decoder/train_cwformer.py#L728):
  CLI default `--max-audio-sec` from 15.0 → 30.0.
- [neural_decoder/inference_cwformer.py:148](neural_decoder/inference_cwformer.py#L148):
  `max_cache_sec` default from 15.0 → 30.0 so inference keeps the full
  training window of context.

---

## Training curriculum changes

Edit [config.py](config.py)'s `create_default_config()` scenarios:

### clean scenario (warmup; current config mostly fine)
- SNR: 15–40 dB (unchanged)
- WPM: **5–50** (was 10–40; extend the low end)
- Augmentations: `agc_probability=0.2` (from 0.3), `qsb_probability=0`,
  `bandpass_probability=0.5` (from 0.5 — unchanged), others off.
- This stage should converge quickly; it's a "learn basic Morse" stage.

### moderate scenario
- SNR: 5–35 dB (was 8–35; slight extension)
- WPM: 5–50 (was 8–45; extend range)
- Augmentations: `agc_probability=0.4` (was 0.5),
  `bandpass_probability=0.6` (was 0.7 — dial back slightly),
  `qsb_probability=0.25`, `qrm_probability=0.15`, `qrn_probability=0.15`,
  `hf_noise_probability=0.3`, `farnsworth_probability=0.2`,
  `multi_op_probability=0.0`.

### full scenario
- **SNR: −5 to 30 dB** (was 3–30). Pushes the model into low-SNR regime.
- WPM: 5–50 (unchanged in spirit; was 5–50).
- Augmentations — dialed back from current:
  - `bandpass_probability: 0.9 → 0.6` so model also sees unfiltered audio
  - `agc_probability: 0.7 → 0.5`
  - `qsb_probability: 0.5` (unchanged)
  - `qrm_probability: 0.3` (unchanged)
  - `qrn_probability: 0.25` (unchanged)
  - `hf_noise_probability: 0.5` (unchanged)
  - `farnsworth_probability: 0.25` (unchanged)
  - `multi_op_probability: 0.15` (unchanged)

**Rationale**: bandpass at 0.9 means almost every training sample has its
spectrum shaped by a radio-style filter. Dropping to 0.6 means 40% of
samples are unfiltered, which teaches the model to generalize. Similarly
for AGC. QRM/QRN/HF noise stay at training distribution — those are
real-world signals.

### Random gain augmentation at input (robustness to caller amplitude)

**Why**: every training sample is peak-normalized to
`target_amplitude ∈ [0.5, 0.9]`, so the model has only ever seen audio
with peak in this narrow range. Real recordings vary wildly. We already
fixed inference to peak-normalize on entry
([neural_decoder/inference_cwformer.py](neural_decoder/inference_cwformer.py),
`_peak_normalize`), but belt-and-suspenders: also teach the model to
tolerate gain variation during training.

**Change**: in `morse_generator.generate_sample()`, after the existing
peak-normalize step
([morse_generator.py:1138-1140](morse_generator.py#L1138-L1140)), apply
a random log-uniform gain:

```python
# Random gain in ±12 dB to teach the model amplitude invariance.
# Applied AFTER peak-normalization so we still have a known reference.
gain_db = rng.uniform(-12.0, 12.0)
gain_lin = 10.0 ** (gain_db / 20.0)
audio_f32 = (audio_f32 * gain_lin).astype(np.float32)
# Re-clip in case gain pushed above 1.0
np.clip(audio_f32, -1.0, 1.0, out=audio_f32)
```

±12 dB (factor of ~0.25 to ~4) covers the realistic range of
uncalibrated radio/SDR captures without pushing too far into clipped
regime. The in-code `np.clip` at ±1 mimics what a DSP chain would do
rather than letting the model see unclipped huge values.

Only apply in `moderate` and `full` scenarios. Keep `clean` at the
original narrow amplitude range so stage 1 remains a gentle warmup.

**Files to change**:
- [morse_generator.py](morse_generator.py): add the gain block at the
  end of `generate_sample()`, conditional on a new config field
  `input_gain_db_range` (default `(0.0, 0.0)` — no gain). Then set the
  range per scenario in `config.py`.
- [config.py](config.py)'s `create_default_config(scenario)` function.
  Each scenario branch needs the above updates, plus
  `input_gain_db_range` per the per-stage discussion.

---

## Streaming correctness fixes

These are standalone bugs independent of the retraining plan — they
should be fixed regardless of whether the retrain happens, because they
affect inference correctness today.

### Fix ONNX streaming mel (A1) — `deploy/inference_onnx.py:162-198`

Two bugs stacked in `MelComputer.compute_streaming`:

1. **Spurious right-pad per chunk**: `audio_padded = np.pad(audio, (0, self.n_fft // 2))`
   adds 200 zeros to the right of every chunk. Training only right-pads
   once, at the end of the whole utterance. Every chunk produces extra
   frames against zeros that don't exist in reality.
2. **Wrong carry-forward buffer**: fixed 240-sample overlap instead of
   `audio[consumed_up_to:]` where `consumed_up_to = n_frames * hop_length`.
   The hop grid breaks as soon as the second chunk arrives.

**Fix**: port the PyTorch `compute_streaming` logic from
[neural_decoder/mel_frontend.py:250-324](neural_decoder/mel_frontend.py#L250-L324)
to the numpy ONNX version. Specifically:
- Remove the `np.pad(audio, (0, self.n_fft // 2))` right-pad.
- Replace the overlap buffer with
  `new_buffer = audio[consumed_up_to:]` where
  `consumed_up_to = n_frames * self.hop`.
- First chunk only: left-pad by `n_fft // 2` if `stft_buffer is None`
  (matches the PyTorch behavior).

**Verify**: after the fix, compute_streaming over a full audio chunk
should produce the same mel frames as the PyTorch path. Write a small
numerical cross-check.

### Mel tail asymmetry (B2) — right-pad on final flush

Training `forward()` at
[neural_decoder/mel_frontend.py:211-212](neural_decoder/mel_frontend.py#L211-L212)
pads both sides: `F.pad(audio, (pad_amount, pad_amount))`.
Streaming `compute_streaming` at
[neural_decoder/mel_frontend.py:278-279](neural_decoder/mel_frontend.py#L278-L279)
left-pads only. For a 15 s clip, training produces 1501 frames,
streaming 1499. The missing 2 tail frames are usually silence, but the
asymmetry:

- Hides tail-of-utterance bugs from `test_streaming_equivalence.py`,
  which silently aligns to the shorter length.
- Causes the last character or two to be decoded with less context
  than training saw.

**Fix**: in `CWFormerStreamingDecoder.flush()`
([neural_decoder/inference_cwformer.py](neural_decoder/inference_cwformer.py)),
append `n_fft // 2 = 200` zeros to the audio buffer before processing
the final chunk. This matches training's right-pad behavior.

Equivalent: add a flag to `compute_streaming()` that signals "this is
the last chunk — right-pad before STFT". Either is fine; the
`flush()`-side fix is less invasive.

**Also update the equivalence test** to not silently align to the
shorter length. It should run both paths until their outputs match in
length; any difference should be a test failure, not a truncation.

### KV-cache trim: trim AFTER append, keep last 750 (B4)

Current trim in
[neural_decoder/cwformer.py:375-381](neural_decoder/cwformer.py#L375-L381)
trims the cache BEFORE concat with new K. After append, the window is
`T_cached + T_chunk` (e.g. 750 + 25 = 775), briefly exceeding the
training max relative distance of 749. Tiny RoPE extrapolation every
chunk.

**Fix**: trim the KV cache AFTER the concat inside `forward_streaming`,
so after each chunk the stored cache is exactly `max_cache_len = 750`.
Only the positions strictly within the training range are ever seen by
attention.

Concretely in
[neural_decoder/cwformer.py](neural_decoder/cwformer.py)'s
`forward_streaming`:

```python
# After encoder returns new_kv_caches:
trimmed = []
for k_cache, v_cache in new_kv_caches:
    # Post-concat cache holds T_prev + T_chunk frames.
    # Keep only the last max_cache_len so attention next chunk
    # sees exactly (max_cache_len + T_chunk) frames, where
    # max_cache_len + T_chunk == training max sequence length.
    if k_cache.shape[2] > max_cache:
        k_cache = k_cache[:, :, -max_cache:, :]
        v_cache = v_cache[:, :, -max_cache:, :]
    trimmed.append((k_cache, v_cache))
```

This is what the code does today. The subtlety is that
`max_cache_len` should be sized as
`training_max_frames - expected_chunk_frames` so that within a chunk's
attention call, the cached + new never exceeds training max.

With the new `max_audio_sec = 30` → training max 1500 frames and typical
500 ms chunks (25 output frames), set:

```python
max_cache_len = 1500 - 25 = 1475
```

Exposed as a config field
([neural_decoder/conformer.py:54](neural_decoder/conformer.py#L54)) —
update the default from 1500 to 1475, and in
[neural_decoder/inference_cwformer.py](neural_decoder/inference_cwformer.py),
expose an optional `max_cache_sec` that computes back from training
audio length.

---

## Doc + cleanup tasks

Low priority but worth doing during the broader refactor:

- **Remove the commitment-delay language** from CLAUDE.md and PLAN.md.
  The code at
  [neural_decoder/inference_cwformer.py:121-125](neural_decoder/inference_cwformer.py#L121-L125)
  correctly argues no delay is needed for causal CTC and just
  re-decodes each chunk. Docs still claim ~2 s delay — purely stale.
- **Fix vocab docstring** at
  [vocab.py:71-72](vocab.py#L71-L72): listed ranges `38–55` and `56–63`
  should be `38–45` (8 punctuation) and `46–51` (6 prosigns). Total 52
  classes.
- **Drop the unused `mask` parameter** from `ConformerEncoder.forward`
  and `ConformerBlock.forward`. Once BN is replaced with LayerNorm,
  there is no correctness reason to build or pass the mask at all
  (causality already prevents valid frames from attending to padded
  frames). Cleaner signatures.

---

## Automated curriculum progression

**Current behavior**: `--scenario` is a CLI arg. User manually runs
`--scenario clean`, then `--scenario moderate --checkpoint ...`, then
`--scenario full --checkpoint ...`. Needs human intervention between
stages.

**Target behavior**: a single invocation with `--auto-curriculum`
advances through `clean → moderate → full` without intervention, saving
a best model per stage and stopping when full converges.

### Convergence criterion

**Plateau detection**: `best_greedy_cer` hasn't improved by more than
`min_delta = 0.003` (0.3 absolute-CER points) for `patience = 25`
consecutive epochs, AND the stage has run at least `min_epochs = 50`.

The `min_epochs` floor prevents premature advancement in `clean` where
CER drops fast but the model isn't fully solidified.

### Advancement mechanics

1. Save current `best_model.pt` as `best_model_{stage}.pt` (e.g.,
   `best_model_clean.pt`, `best_model_moderate.pt`).
2. Transition to next scenario:
   - `clean → moderate → full → stop`.
3. On transition:
   - Load the saved `best_model_{stage}.pt` as the starting checkpoint
     for the next stage.
   - Keep optimizer momentum buffers.
   - Reset the LR schedule to a fresh warmup for the remaining epochs
     (fresh cosine from peak → 5% floor, not `--lr-resume`).
   - Reset `best_greedy_cer` to `inf` so the new stage's best is tracked
     from scratch.
   - Reset the plateau counter.
4. When the `full` stage plateaus, write a final marker file (e.g.,
   `training_complete.txt`) and exit cleanly.

### New CLI flag

```
--auto-curriculum          # enables the above. Default OFF for
                           # backwards compatibility.
--curriculum-patience 25   # plateau patience (epochs)
--curriculum-min-delta 0.003
--curriculum-min-epochs 50
```

### Epoch budget

With `--auto-curriculum`, total epochs are spread across stages. A
reasonable budget:
- `clean`: up to 150 epochs (typically converges in ~80–100)
- `moderate`: up to 200 epochs (typically converges in ~120–150)
- `full`: up to 300 epochs (typically converges in ~200+)

Total ceiling: 650 epochs. In practice, auto-advance will cut short
stages that plateau early.

Override with `--epochs` to set the total ceiling; `--auto-curriculum`
will distribute it proportionally (`clean=25%, moderate=30%, full=45%`).

### Files to change

- [neural_decoder/train_cwformer.py](neural_decoder/train_cwformer.py) —
  main training loop. Add CLI flags, plateau tracking, and the transition
  logic after validation each epoch.

---

## Benchmark script changes

### Long audio by default

Current `_base_config` sets `min_chars=10, max_chars=18` (I shortened
these during debugging to keep audio ≤ 15 s). Revert and set so that
**every benchmark sample exceeds the training `max_audio_sec=30`**:

- At 35 WPM (highest tested): 30 s × 2.92 chars/s = **88 chars minimum**.
  Round up to `min_chars = 90`.
- At 15 WPM (lowest tested): `max_chars` upper bound should keep audio
  reasonable. `max_chars = 150` gives 120 s at 15 WPM, 51 s at 35 WPM.
- `max_duration_sec=None` passed to `generate_sample` so no cap.

This stretches the model past training distribution on purpose — that's
the real-world test. Inference must handle it via streaming + KV cache
cycling without falling apart.

### Phase 1 with 100% bandpass

User-specified: Phase 1 should always apply a bandpass filter between
250–600 Hz (typical amateur CW receiver filter range). Concrete config:

```python
mc.bandpass_probability = 1.0
mc.bandpass_bw_min = 250.0
mc.bandpass_bw_max = 600.0
mc.bandpass_order_min = 4
mc.bandpass_order_max = 8
```

This matches a realistic radio receiver setup. The rest of Phase 1
augmentations stay off.

### Phase 2 baseline

Phase 2's baseline row reads from Phase 1 results. Since Phase 1 now
includes bandpass, Phase 2's baseline will reflect bandpass-on. That's
consistent — Phase 2 measures marginal effect of *additional*
augmentations on top of a realistic baseline.

Phase 2 should NOT add another bandpass on top of the baseline's
bandpass (avoid double-filtering). Remove or update the `BP 250 Hz` and
`BP 400 Hz` rows in `AUGMENTATIONS`.

### Files to change

- [benchmark_cwformer.py](benchmark_cwformer.py) — update `_base_config`,
  remove the extra BP rows from Phase 2's `AUGMENTATIONS` list, make sure
  `generate_sample` is called WITHOUT `max_duration_sec` (we want long
  audio in the benchmark).

---

## Execution plan

Run these as four agents. Agents 2 and 3 touch files no other agent
touches and can run fully in parallel. Agents 1 and 4 both touch
`neural_decoder/conformer.py` (different functions, but same file), so
either (a) run them in sequence, or (b) run each in a git worktree
(isolation="worktree") and merge sequentially. The different-function
edits should auto-merge cleanly.

Suggested launch: spawn Agent 2 and Agent 3 in parallel in the main
checkout; spawn Agent 1 and Agent 4 each in their own worktree. When
all return, merge Agent 1's worktree first (it has the larger
architectural changes), then rebase Agent 4's worktree on top.

### Agent 1: Model architecture + training data generation

**Task**: update the model defaults (kernel, cache size, LayerNorm
swap) and augment the data generator with random input gain.

**Files to edit**:
- [neural_decoder/conformer.py](neural_decoder/conformer.py):
  - `ConformerConfig.conv_kernel` 31 → 63.
  - `ConformerConfig.max_cache_len` 1500 → 1475.
  - `ConvolutionModule`: replace `nn.BatchNorm1d(d_model)` with
    `nn.LayerNorm(d_model)`. Re-arrange transpose(s) as needed so the
    norm sees `(B, T, D)`.
- [neural_decoder/train_cwformer.py](neural_decoder/train_cwformer.py):
  CLI defaults for `--conv-kernel` (31 → 63) and `--max-audio-sec`
  (15.0 → 30.0).
- [neural_decoder/inference_cwformer.py](neural_decoder/inference_cwformer.py):
  `max_cache_sec` default 15.0 → 30.0 in `CWFormerStreamingDecoder.__init__`.
- [config.py](config.py): update each scenario's
  SNR/WPM/augmentation-probability values per the "Training curriculum
  changes" section. Also add `input_gain_db_range` field to
  `MorseConfig` (default `(0.0, 0.0)`) and set it per scenario:
  `clean=(0, 0)`, `moderate=(-6, 6)`, `full=(-12, 12)`.
- [morse_generator.py](morse_generator.py): add the random-gain block
  at the end of `generate_sample()` (after peak-normalize), conditional
  on `config.input_gain_db_range != (0.0, 0.0)`.

**Verify**:
- `python -c "from neural_decoder.cwformer import CWFormer, CWFormerConfig; print(CWFormer(CWFormerConfig()).num_params)"`
  — parameter count should be ~19.5M + ~100K (wider conv) − small
  (LayerNorm has fewer params than BN since no running stats).
- `python -c "from morse_generator import generate_sample; from config import create_default_config; import numpy as np; rng=np.random.default_rng(0); audio, _, _ = generate_sample(create_default_config('full').morse, rng=rng); print(audio.shape, float(np.max(np.abs(audio))))"`
  — should produce audio and peak shouldn't always be 0.7 now.

### Agent 2: Auto-curriculum training logic

### Agent 2: Auto-curriculum training logic

**Task**: add automatic curriculum progression to the training loop.

**Files to edit**:
- [neural_decoder/train_cwformer.py](neural_decoder/train_cwformer.py):
  - Add CLI flags: `--auto-curriculum`, `--curriculum-patience`,
    `--curriculum-min-delta`, `--curriculum-min-epochs`.
  - In the training loop, after val_results each epoch:
    - If `--auto-curriculum`: track epochs-since-improvement. When
      plateau detected AND min_epochs hit AND current scenario isn't
      `full`: save `best_model_{stage}.pt`, advance scenario, re-init
      config/scheduler, reset best tracker and plateau counter, reload
      model from saved best.
    - If current scenario IS `full` and plateau detected: exit cleanly.
  - When loading model for new stage, preserve optimizer state but reset
    LR schedule to a fresh cosine over the remaining epochs.
  - Print a clear stage-transition banner to stdout for log grep.

**Keep backward compatibility**: without `--auto-curriculum`, existing
behavior (single `--scenario` per run) is preserved.

**Verify**: run a quick smoke test
```
python -m neural_decoder.train_cwformer --scenario clean --auto-curriculum --epochs 10 --curriculum-min-epochs 3 --curriculum-patience 3 --ckpt-dir /tmp/smoke
```
Confirm it advances through scenarios in the log output without crashing.

### Agent 3: Benchmark script update

**Task**: update `benchmark_cwformer.py` to test long-audio, bandpass-on
scenarios.

**Files to edit**:
- [benchmark_cwformer.py](benchmark_cwformer.py):
  - `_base_config`:
    - `min_chars = 90`, `max_chars = 150`.
    - `bandpass_probability = 1.0`, `bandpass_bw_min = 250.0`,
      `bandpass_bw_max = 600.0`, `bandpass_order_min = 4`,
      `bandpass_order_max = 8`.
    - Other augmentation probabilities stay at 0.
  - Remove `BP 250 Hz` and `BP 400 Hz` entries from the `AUGMENTATIONS`
    list (they're now baseline, not additional).
  - In `eval_cell`, call `generate_sample(mc, rng=rng)` *without*
    `max_duration_sec=15.0` — we want long audio.

**Verify**: `python benchmark_cwformer.py --help` should still work; a
dry run of one cell with a small `--samples` should produce audio > 30 s
at 35 WPM.

### Agent 4: Streaming correctness fixes + cleanups

**Task**: fix the standalone bugs independent of the retrain. These are
wins today even without the new model.

**Files to edit**:
- [deploy/inference_onnx.py](deploy/inference_onnx.py): rewrite
  `MelComputer.compute_streaming` to match the PyTorch logic in
  [neural_decoder/mel_frontend.py:250-324](neural_decoder/mel_frontend.py#L250-L324).
  No per-chunk right-pad; carry forward `audio[consumed_up_to:]` where
  `consumed_up_to = n_frames * hop`; left-pad only on the first chunk.
- [neural_decoder/inference_cwformer.py](neural_decoder/inference_cwformer.py):
  in `flush()`, right-pad the audio buffer with `n_fft // 2` (200)
  zeros before processing the final chunk, so the tail frame count
  matches training.
- [neural_decoder/cwformer.py](neural_decoder/cwformer.py): update
  KV-cache trimming comment/logic so the post-concat trim always
  yields exactly `max_cache_len` frames; verify it's trimming after
  concat, not before.
- [tests/test_streaming_equivalence.py](tests/test_streaming_equivalence.py):
  update the alignment so it doesn't silently truncate to the shorter
  length. If `flush()` is fixed to right-pad, the lengths should now
  match; any remaining mismatch is a real bug.
- [CLAUDE.md](CLAUDE.md), [PLAN.md](PLAN.md): remove the
  commitment-delay language (the code correctly doesn't use one).
- [vocab.py](vocab.py): fix the docstring at lines 71-72 — punctuation
  indices are 38–45, prosigns are 46–51 (total 52).
- [neural_decoder/conformer.py](neural_decoder/conformer.py) and
  [neural_decoder/cwformer.py](neural_decoder/cwformer.py): drop the
  unused `mask` parameter from `ConformerEncoder.forward`,
  `ConformerBlock.forward`, and `CWFormer.forward`'s mask computation.
  Causality + LayerNorm (after Agent 1) eliminates the need for
  explicit padding masking.

**Verify**:
- Run `python tests/test_streaming_equivalence.py --checkpoint
  checkpoints_full/best_model.pt --wpm 35` (using the old model, before
  retrain) and confirm tail-frame alignment no longer silently
  truncates. The test should report exactly matching tensor shapes on
  the final chunk.
- ONNX cross-check: feed a short clip through both the PyTorch and
  ONNX paths and diff log_probs. Should match to fp32 precision.

---

## Training command for the new model

After agents finish, start training with:

```
python -m neural_decoder.train_cwformer --scenario clean --auto-curriculum --ckpt-dir checkpoints_v2 --workers 10 --epochs 600 --confidence-penalty 0.1
```

Confidence penalty 0.1 is the value that was active during the last run
and didn't hurt; keep it as a mild regularization against late-stage
overconfidence.

Expected behavior:
- `clean` stage converges around epoch 80–120 → advance.
- `moderate` stage converges around epoch 250–350 → advance.
- `full` stage runs to epoch 500–600 → plateau-exit or hit the cap.
- Three `best_model_{stage}.pt` saves plus final `best_model.pt` (copy
  of the `full` best).

---

## Success criteria

These are the numbers to hit after full-stage convergence. If the model
lands here, the plan worked; if not, come back and iterate.

1. **Training val CER (full scenario, augmented)**: ≤ 5% (was ~7%).
2. **Benchmark Phase 1 CER average (long audio + bandpass)**: ≤ 8%.
   Specifically:
   - SNR ≥ 10, WPM ≥ 25: ≤ 3%.
   - SNR ≥ 10, WPM = 15: ≤ 10%. (Low WPM is the stretch target — the
     wider conv kernel plus longer training audio should make this
     achievable.)
   - SNR = −5, WPM ≥ 25: ≤ 25%. (Low SNR is OOD-stretch; if we hit this
     the distribution extension worked.)
3. **Streaming equivalence test** with in-distribution audio (e.g.
   `--wpm 25`, ≤ 30 s) still shows `block0_after_mha_out diff < 1e-3`.
4. **Phase 3 context ramp-up**: error rate at character position 1
   should be ≤ 5% and drop below steady-state by position 10. This
   measures the model's ability to bootstrap from a cold KV cache.

---

## Risks and things to watch

- **Training time doubles roughly** from the 30 s max audio (attention
  is O(T²)). If a training epoch that currently takes ~5 min balloons
  past 15 min, reduce `samples_per_epoch` or drop batch_size and use
  gradient accumulation.
- **Wider conv kernel slows convolution ~2× in the conv module**. Total
  impact on step time: modest (conv is not the dominant op; attention
  is), probably +10-15%.
- **Auto-curriculum's patience tuning is empirical.** 25 epochs might be
  too aggressive or too conservative. Watch the first run's logs and
  adjust if a stage advances too early (CER spikes on the new stage
  indefinitely) or too late (CER plateaued long ago, training just
  wastes time). `--curriculum-patience` is the knob.
- **The old checkpoint is shape-incompatible** with the new conv kernel.
  Don't try to warm-start from `checkpoints_full/best_model.pt` —
  start fresh in `checkpoints_v2/`.
- **RoPE table**: auto-extending handles whatever position range shows
  up at inference. No explicit change needed; trust the existing
  implementation in [rope.py](neural_decoder/rope.py).
- **Normalization at inference**: already added to `decode_audio` in the
  last session. Don't re-add it. `feed_audio` intentionally doesn't
  normalize (caller is responsible for live audio gain).
- **LayerNorm vs BatchNorm training dynamics differ slightly**: learning
  rate might want a small tweak. LayerNorm is generally more stable at
  higher LR, so the existing peak LR 3e-4 should be fine or slightly
  conservative. If training diverges early, halve LR first before
  declaring a problem. Otherwise expected to behave very similarly in
  convergence speed and final loss — LayerNorm in the Conformer conv
  module is well-studied (Google streaming Conformer, NVIDIA
  FastConformer both use it).
- **Random-gain augmentation range ±12 dB is a starting point.** If the
  model underfits on quiet audio, widen to ±20 dB. If it struggles with
  the clip-at-±1 pushing it into hard nonlinearity, narrow to ±6 dB.
  First check by running Phase 1 at several fixed gain levels after
  training.

---

## What NOT to change

These were considered and explicitly rejected:

- **Don't re-implement relative-RoPE caching** (raw-K cache, rotate
  every chunk). Was tried, made numerical diff worse with no correctness
  benefit. Absolute-position RoPE with growing `pos_offset` is correct.
- **Don't add beam search or LM at decode time.** Greedy CTC only; beam
  search was tested and didn't help.
- **Don't enable augmentations in Phase 2's baseline beyond what
  Phase 1 now has.** Phase 2 measures *marginal* effect of added
  augmentations on top of the realistic baseline.
- **Don't widen `d_model` or add layers** as a first response to
  remaining low-WPM weakness. The conv kernel + longer training audio
  should be tried first; capacity changes are a separate experiment.

---

## Where to read next if confused

- [CLAUDE.md](CLAUDE.md) — project overview and architecture reference.
- [config.py](config.py) — the scenario config functions.
- [neural_decoder/conformer.py](neural_decoder/conformer.py) — Conformer
  block with causal attention, conv module, MHA with RoPE.
- [neural_decoder/cwformer.py](neural_decoder/cwformer.py) — main model
  with `forward()` (training) and `forward_streaming()` (inference).
- [neural_decoder/train_cwformer.py](neural_decoder/train_cwformer.py) —
  training loop (where auto-curriculum lives).
- [benchmark_cwformer.py](benchmark_cwformer.py) — evaluation grid.
- [tests/test_streaming_equivalence.py](tests/test_streaming_equivalence.py) —
  sanity check for streaming-path correctness (run after code changes).

---

## Memory references from prior work (do not duplicate)

- Greedy CTC only; beam search + LM don't help.
- User's shell is PowerShell on Windows; give commands as single-line or
  with backtick continuation, never backslash.

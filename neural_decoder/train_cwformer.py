#!/usr/bin/env python3
"""
train_cwformer.py — Training loop for the causal streaming CW-Former.

The model uses fully causal attention (is_causal=True) and causal
convolutions during training. No explicit mask construction is needed —
causality is enforced internally by the model architecture. Weights are
shape-compatible with the original bidirectional CW-Former for fine-tuning.

Usage:
    # Quick test (verify pipeline)
    python -m neural_decoder.train_cwformer --scenario test

    # Stage 1: Clean conditions
    python -m neural_decoder.train_cwformer --scenario clean

    # Stage 2: Resume from clean, moderate augmentations
    python -m neural_decoder.train_cwformer --scenario moderate \
        --checkpoint checkpoints_cwformer/best_model.pt

    # Stage 3: Resume from moderate, full augmentations
    python -m neural_decoder.train_cwformer --scenario full \
        --checkpoint checkpoints_cwformer/best_model_moderate.pt

    # Fine-tune from bidirectional CWNet checkpoint
    python -m neural_decoder.train_cwformer --scenario full \
        --checkpoint /path/to/cwnet_bidirectional/best_model.pt
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

import vocab as vocab_module
from config import Config, create_default_config
from neural_decoder.cwformer import CWFormer, CWFormerConfig
from neural_decoder.conformer import ConformerConfig
from neural_decoder.mel_frontend import MelFrontendConfig
from neural_decoder.dataset_audio import AudioDataset, collate_fn
from neural_decoder.inference_cwformer import CWFormerStreamingDecoder


# ---------------------------------------------------------------------------
# CTC decoding helpers
# ---------------------------------------------------------------------------

def greedy_decode(log_probs: torch.Tensor) -> str:
    return vocab_module.decode_ctc(log_probs, blank_idx=0, strip_trailing_space=True)


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
    # Strip boundary spaces — the model is trained with [space]+text+[space]
    # targets but the reference text does not include boundary tokens.
    h = hypothesis.strip().upper()
    r = reference.strip().upper()
    if not r:
        return 0.0 if not h else 1.0
    return levenshtein(h, r) / len(r)


# ---------------------------------------------------------------------------
# Buffer pre-generation
# ---------------------------------------------------------------------------

def generate_disk_cache(
    dataset: "AudioDataset",
    micro_batch: int,
    num_workers: int,
    buffer_epochs: int = 1,
    cache_dir: str = "",
    buffer_gen: int = 0,
) -> list:
    """Pre-generate buffer_epochs × epoch_size samples and save to disk.

    Returns a list of Path objects (file paths to .pt batch files), NOT
    loaded tensors.  The training loop loads batches lazily during iteration
    to keep RAM usage constant regardless of buffer size.
    """
    loader = DataLoader(
        dataset, batch_size=micro_batch, collate_fn=collate_fn,
        num_workers=num_workers, pin_memory=False,
        prefetch_factor=4 if num_workers > 0 else None,
        persistent_workers=False,
    )

    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    index_file = cache_path / f"gen{buffer_gen}_index.pt"

    if index_file.exists():
        file_list = torch.load(index_file, weights_only=False)
        print(f"  Reusing cached buffer: {len(file_list)} batches from {cache_dir}",
              file=sys.stderr)
        return file_list

    file_list = []
    batch_idx = 0
    for pass_idx in range(buffer_epochs):
        desc = (f"Caching buffer pass {pass_idx + 1}/{buffer_epochs}"
                if buffer_epochs > 1 else "Caching buffer")
        for batch in tqdm(loader, desc=desc, file=sys.stderr, leave=False):
            p = cache_path / f"gen{buffer_gen}_batch{batch_idx:06d}.pt"
            torch.save(batch, p)
            file_list.append(p)
            batch_idx += 1
    torch.save(file_list, index_file)
    return file_list


def _lazy_disk_iter(file_paths: list):
    """Yield batches by loading one .pt file at a time from disk."""
    for p in file_paths:
        yield torch.load(p, weights_only=False)


def generate_epoch_buffer(
    dataset: "AudioDataset",
    micro_batch: int,
    num_workers: int,
    buffer_epochs: int = 1,
) -> list:
    """Pre-generate buffer_epochs × epoch_size samples into a list of batches (in-memory)."""
    loader = DataLoader(
        dataset, batch_size=micro_batch, collate_fn=collate_fn,
        num_workers=num_workers, pin_memory=False,
        prefetch_factor=4 if num_workers > 0 else None,
        persistent_workers=False,
    )

    buffer = []
    for pass_idx in range(buffer_epochs):
        desc = (f"Filling buffer pass {pass_idx + 1}/{buffer_epochs}"
                if buffer_epochs > 1 else "Filling buffer")
        for batch in tqdm(loader, desc=desc, file=sys.stderr, leave=False):
            buffer.append(batch)
    return buffer


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(
    model: CWFormer,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool = False,
    stream_decoder: Optional[CWFormerStreamingDecoder] = None,
    stream_max_audio_sec: Optional[float] = None,
    stream_max_samples: Optional[int] = None,
) -> dict:
    """Evaluate model on a dataset.

    Returns dict with loss, greedy_cer, and (when a ``stream_decoder``
    is provided) ``stream_cer`` plus ``stream_n``.
    The streaming path re-runs each val sample through
    ``CWFormerStreamingDecoder`` so the reported number tracks the
    actual deployment-time chunk-by-chunk CER. It is expected to agree
    with ``greedy_cer`` to within ~1% absolute on converged in-
    distribution audio; persistent drift beyond that is a regression
    signal against the streaming state-carry code (or cuDNN kernel
    heuristics diverging between full-sequence and short-chunk inputs).

    The streaming decoder calls ``feed_audio`` + ``flush`` directly on
    the already-generated val audio instead of ``decode_audio``: val
    samples come from ``morse_generator``, which peak-normalises each
    sample; re-normalising in ``decode_audio`` would scale the audio
    away from what ``model.forward()`` sees in the same loop and
    inflate any apparent divergence between the two metrics.
    """
    model.eval()
    ctc_loss_fn = nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)

    total_loss = 0.0
    total_batches = 0
    all_cer_greedy = []
    all_cer_stream: list = []
    stream_skipped = 0

    with torch.no_grad():
        for audio, targets, audio_lens, target_lens, texts in loader:
            audio = audio.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            audio_lens = audio_lens.to(device, non_blocking=True)
            target_lens = target_lens.to(device, non_blocking=True)

            with autocast("cuda", enabled=use_amp):
                log_probs, out_lens = model(audio, audio_lens)

                # CTC feasibility: skip samples where output is too short
                valid = out_lens >= target_lens
                if not valid.all():
                    idx = valid.nonzero(as_tuple=True)[0]
                    if len(idx) == 0:
                        del audio, targets, audio_lens, target_lens, log_probs, out_lens
                        continue
                    keep_idx = idx.cpu().tolist()
                    log_probs = log_probs[:, idx, :]
                    targets = targets[idx]
                    out_lens = out_lens[idx]
                    target_lens = target_lens[idx]
                    texts = [texts[i] for i in keep_idx]
                    audio_lens_kept = audio_lens[idx]
                    audio_kept = audio[idx]
                else:
                    audio_lens_kept = audio_lens
                    audio_kept = audio

                loss = ctc_loss_fn(log_probs, targets, out_lens, target_lens)

            total_loss += loss.item()
            total_batches += 1

            # Move to CPU for CER computation, free GPU memory
            log_probs_cpu = log_probs.cpu()
            out_lens_cpu = out_lens.cpu()

            # Per-sample audio needed for the streaming pass; stash on CPU
            # so we can release GPU audio before streaming inference runs
            # (streaming decoder is on the same device, but its own audio
            # tensors will be small per-chunk transfers).
            stream_inputs: list = []
            # Per-sample streaming is serial and ~2 orders of magnitude
            # slower than batched full-forward; stop collecting for stream
                # once we've hit the cap so the rest of validation
            # (full-forward CER + CTC loss) still runs on all samples.
            stream_budget_remaining = (
                stream_decoder is not None and (
                    stream_max_samples is None
                    or len(all_cer_stream) + stream_skipped < stream_max_samples
                )
            )
            if stream_budget_remaining:
                audio_cpu = audio_kept.detach().cpu().numpy()
                audio_lens_cpu = audio_lens_kept.detach().cpu().tolist()
                for i, L in enumerate(audio_lens_cpu):
                    L = int(L)
                    stream_inputs.append(audio_cpu[i, :L])

            del audio, audio_kept, targets, audio_lens, audio_lens_kept
            del target_lens, log_probs, out_lens, loss

            B = log_probs_cpu.shape[1]
            for i in range(B):
                T_i = int(out_lens_cpu[i].item())
                lp_i = log_probs_cpu[:T_i, i, :]

                hyp_greedy = greedy_decode(lp_i)
                cer_g = compute_cer(hyp_greedy, texts[i])
                all_cer_greedy.append(cer_g)

            del log_probs_cpu, out_lens_cpu

            if stream_decoder is not None and stream_inputs:
                sample_rate = stream_decoder.sample_rate
                for i, audio_i in enumerate(stream_inputs):
                    if (stream_max_samples is not None
                            and len(all_cer_stream) >= stream_max_samples):
                        break
                    # Skip anything longer than the KV cache window so the
                    # streaming path doesn't start trimming context the
                    # full-forward path still has.
                    if stream_max_audio_sec is not None:
                        if len(audio_i) > stream_max_audio_sec * sample_rate:
                            stream_skipped += 1
                            continue
                    stream_decoder.reset()
                    stream_decoder.feed_audio(audio_i)
                    stream_decoder.flush()
                    hyp_stream = stream_decoder.get_full_text()
                    cer_s = compute_cer(hyp_stream, texts[i])
                    all_cer_stream.append(cer_s)
                # Release accumulated log-probs from the last sample.
                stream_decoder.reset()

            del stream_inputs

    results = {
        "loss": total_loss / max(1, total_batches),
        "greedy_cer": float(np.mean(all_cer_greedy)) if all_cer_greedy else 1.0,
    }
    if stream_decoder is not None:
        results["stream_cer"] = (
            float(np.mean(all_cer_stream)) if all_cer_stream else float("nan")
        )
        results["stream_n"] = len(all_cer_stream)
        results["stream_skipped"] = stream_skipped
    return results


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_rocm = hasattr(torch.version, 'hip') and torch.version.hip is not None
    if args.no_amp or is_rocm:
        use_amp = False
    else:
        use_amp = device.type == "cuda"
    # pin_memory is counter-productive on unified memory (e.g. AMD APU / iGPU)
    use_pin_memory = device.type == "cuda" and not is_rocm
    if is_rocm:
        print(f"Device: {device} (ROCm {torch.version.hip}), AMP disabled, pin_memory disabled")
    else:
        print(f"Device: {device}, AMP: {use_amp}")

    if device.type == "cuda":
        # Auto-tune convolution algorithms for consistent input sizes
        torch.backends.cudnn.benchmark = True
        # Enable TF32 on Ampere+ for faster fp32 matmuls via tensor cores
        if not is_rocm:
            torch.set_float32_matmul_precision('high')

    # ---- Auto-curriculum setup ----
    # When --auto-curriculum is set, distribute the total epoch budget across
    # the three stages (clean/moderate/full) proportionally. Each stage still
    # plateau-exits early if converged. If the user didn't supply --epochs,
    # default to a 650-epoch ceiling.
    CURRICULUM_ORDER = ("clean", "moderate", "full")
    CURRICULUM_FRACTIONS = {"clean": 0.25, "moderate": 0.30, "full": 0.45}
    current_scenario = args.scenario

    if args.auto_curriculum:
        if current_scenario not in CURRICULUM_ORDER:
            print(f"WARNING: --auto-curriculum requires starting scenario in "
                  f"{CURRICULUM_ORDER}; got '{current_scenario}'. Falling back "
                  f"to 'clean'.", file=sys.stderr)
            current_scenario = "clean"
        overall_budget = args.epochs if args.epochs is not None else 650
        # Fraction of the overall budget allocated to each stage.  Unused
        # stage allocations (e.g. if we start mid-curriculum) are ignored.
        stage_budgets = {
            s: max(1, int(round(overall_budget * CURRICULUM_FRACTIONS[s])))
            for s in CURRICULUM_ORDER
        }
        print(f"Auto-curriculum enabled. Overall budget: {overall_budget} epochs. "
              f"Per-stage caps: "
              + ", ".join(f"{s}={stage_budgets[s]}" for s in CURRICULUM_ORDER))
        print(f"  patience={args.curriculum_patience}, "
              f"min_delta={args.curriculum_min_delta}, "
              f"min_epochs={args.curriculum_min_epochs}")

    # ---- Config ----
    config = create_default_config(current_scenario)

    if args.auto_curriculum:
        # Per-stage epoch cap overrides config.training.num_epochs.
        config.training.num_epochs = stage_budgets[current_scenario]
    elif args.epochs is not None:
        config.training.num_epochs = args.epochs

    # ---- Model config ----
    mel_cfg = MelFrontendConfig(
        sample_rate=config.morse.sample_rate,
        spec_augment=True,
    )
    conformer_cfg = ConformerConfig(
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        conv_kernel=args.conv_kernel,
        dropout=args.dropout,
    )
    model_cfg = CWFormerConfig(
        mel=mel_cfg,
        conformer=conformer_cfg,
    )

    # ---- Checkpoint directory ----
    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config_path = ckpt_dir / "config.json"
    config.save(str(config_path))

    # ---- Model ----
    model = CWFormer(model_cfg).to(device)
    print(f"CW-Former (causal streaming): {model.num_params:,} parameters")
    print(f"  d_model={conformer_cfg.d_model}, n_heads={conformer_cfg.n_heads}, "
          f"n_layers={conformer_cfg.n_layers}, d_ff={conformer_cfg.d_ff}, "
          f"conv_kernel={conformer_cfg.conv_kernel}")
    print(f"  subsample=2x (20ms/frame), causal attention + causal conv")

    # ---- Load checkpoint if resuming ----
    start_epoch = 0
    best_val_loss = float("inf")
    best_greedy_cer = float("inf")
    ckpt = None
    if args.checkpoint and Path(args.checkpoint).exists():
        print(f"Loading checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        if "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"], strict=False)
        else:
            model.load_state_dict(ckpt, strict=False)
        if "epoch" in ckpt:
            start_epoch = ckpt["epoch"] + 1
        prev_scenario = ckpt.get("scenario", "")
        if prev_scenario == current_scenario:
            best_val_loss = ckpt.get("best_val_loss", float("inf"))
            # best_greedy_cer is a new field; fall back to inf so the next
            # eval registers a new best selected by CER regardless of what
            # the old best_model.pt held (it was chosen by val_loss).
            best_greedy_cer = ckpt.get("best_greedy_cer", float("inf"))
        else:
            if prev_scenario:
                print(f"  Scenario changed ({prev_scenario} -> {current_scenario}), "
                      f"resetting best metrics")
        print(f"  Resuming from epoch {start_epoch}, "
              f"best_val_loss={best_val_loss:.4f}, "
              f"best_greedy_cer={best_greedy_cer:.4f}")

    # ---- Optimizer + scheduler ----
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.01,
        betas=(0.9, 0.98),
        eps=1e-9,
    )
    total_epochs = config.training.num_epochs

    if args.lr_resume and ckpt is not None:
        # ---- Continue LR schedule from checkpoint ----
        # Restore optimizer state first (momentum buffers + saved LR)
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])

        warmup_epochs = min(5, max(1, total_epochs // 40))
        lr_floor = args.lr_floor

        def lr_lambda(epoch: int) -> float:
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return max(lr_floor, cosine)

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        # Restore scheduler position on the cosine curve
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])

        print(f"  LR resumed at {optimizer.param_groups[0]['lr']:.2e} "
              f"(epoch {start_epoch}/{total_epochs})")
    else:
        # ---- Fresh LR schedule over remaining epochs ----
        # Restore optimizer momentum buffers but reset LR to args.lr
        if ckpt is not None and "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            for pg in optimizer.param_groups:
                pg["lr"] = args.lr
                pg.pop("initial_lr", None)

        remaining_epochs = total_epochs - start_epoch
        warmup_epochs = min(5, max(1, remaining_epochs // 40))
        lr_floor = args.lr_floor

        def lr_lambda(epoch: int) -> float:
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            progress = (epoch - warmup_epochs) / max(1, remaining_epochs - warmup_epochs)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return max(lr_floor, cosine)

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        if start_epoch > 0:
            print(f"  Fresh LR schedule: {remaining_epochs} remaining epochs, "
                  f"peak lr={args.lr:.2e}")

    scaler = GradScaler("cuda", enabled=use_amp)

    # ---- Loss ----
    ctc_loss_fn = nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)

    # ---- Datasets ----
    # Audio generation is CPU-bound — use smaller batches and fewer samples
    samples_per_epoch = min(config.training.samples_per_epoch, 20000)
    val_samples = min(config.training.val_samples, 5000)

    micro_batch = args.batch_size
    effective_batch = 64  # Target effective batch for audio
    accum_steps = max(1, effective_batch // micro_batch)

    num_workers = args.workers
    reuse_factor = args.reuse_factor

    def _build_dataloaders(cfg: Config):
        """Construct train/val datasets and loaders for the given scenario cfg.

        Called once up-front and again on each auto-curriculum stage advance so
        the new scenario's MorseConfig (SNR, WPM, augmentations) takes effect.
        Returns (train_ds, val_ds, train_loader or None, val_loader).
        train_loader is None when reuse_factor > 1 (the buffer path owns its
        own loader inside the epoch body).
        """
        _train_ds = AudioDataset(
            cfg, epoch_size=samples_per_epoch, seed=None,
            qso_text_ratio=0.5, max_audio_sec=args.max_audio_sec,
        )
        _val_ds = AudioDataset(
            cfg, epoch_size=val_samples, seed=None,  # fresh samples each eval
            qso_text_ratio=0.5, max_audio_sec=args.max_audio_sec,
        )
        _train_loader = None
        if reuse_factor <= 1:
            _train_loader = DataLoader(
                _train_ds, batch_size=micro_batch, collate_fn=collate_fn,
                num_workers=num_workers, pin_memory=use_pin_memory,
                prefetch_factor=4 if num_workers > 0 else None,
                persistent_workers=num_workers > 0,
            )
        _val_loader = DataLoader(
            _val_ds, batch_size=micro_batch, collate_fn=collate_fn,
            num_workers=min(num_workers, 4), pin_memory=use_pin_memory,
            prefetch_factor=4 if num_workers > 0 else None,
            persistent_workers=min(num_workers, 4) > 0,
        )
        return _train_ds, _val_ds, _train_loader, _val_loader

    train_ds, val_ds, train_loader, val_loader = _build_dataloaders(config)

    # ---- Streaming-mode val decoder ----
    # 0 disables the feature (no decoder built, no per-epoch overhead).
    stream_val_every = args.stream_val_every_n_epochs
    stream_chunk_ms = args.stream_val_chunk_ms
    stream_max_cache_sec = args.stream_val_max_cache_sec
    stream_max_samples = args.stream_val_samples
    stream_decoder: Optional[CWFormerStreamingDecoder] = None
    if stream_val_every > 0:
        stream_decoder = CWFormerStreamingDecoder.from_model(
            model=model,
            model_cfg=model_cfg,
            sample_rate=mel_cfg.sample_rate,
            chunk_ms=stream_chunk_ms,
            device=device,
            max_cache_sec=stream_max_cache_sec,
        )
        print(f"Streaming-val enabled: every {stream_val_every} epoch(s), "
              f"chunk_ms={stream_chunk_ms}, max_cache_sec={stream_max_cache_sec}, "
              f"max_samples={stream_max_samples}")

    # ---- CSV log ----
    log_path = ckpt_dir / "training_log.csv"
    log_fields = ["epoch", "train_loss", "train_entropy", "val_loss",
                  "greedy_cer", "stream_cer", "lr", "time_s"]
    if not log_path.exists() or start_epoch == 0:
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(log_fields)

    buffer_epochs = args.buffer_epochs
    cache_dir = args.cache_dir

    # ---- Training loop ----
    reuse_str = (f", buffer_epochs={buffer_epochs}, reuse_factor={reuse_factor}"
                 if reuse_factor > 1 else "")
    print(f"\nTraining: {total_epochs} epochs, {samples_per_epoch} samples/epoch, "
          f"micro_batch={micro_batch}, accum={accum_steps} (effective={micro_batch*accum_steps}), "
          f"workers={num_workers}{reuse_str}")
    print(f"Scenario: {current_scenario}"
          + (f", cache_dir={cache_dir}" if cache_dir else ""))

    # Buffer state for reuse_factor > 1.
    #
    # In-memory path (cache_dir=None):
    #   Fill phase: generate one epoch at a time, train on fresh data only.
    #   Replay does not start until buffer_epochs fill epochs are done.
    #   Replay phase: shuffle full buffer, slice to one epoch's worth of batches.
    #
    # Disk-cache path (cache_dir set):
    #   Blocking fill: generates all buffer_epochs passes up front, saves to disk.
    #   GPU is idle during fill but disk enables buffers too large for RAM.
    if reuse_factor > 1 and reuse_factor <= buffer_epochs:
        print(f"WARNING: reuse_factor ({reuse_factor}) <= buffer_epochs "
              f"({buffer_epochs}). No replay will occur. "
              f"Increase --reuse-factor above --buffer-epochs.", file=sys.stderr)
    _buffer: list = []
    _buffer_rng = np.random.default_rng(99)
    # In-memory state
    _phase: str = "fill"
    _fill_count: int = 0
    _replay_count: int = 0
    _batches_per_epoch: int = 0
    # Disk-cache state
    _buffer_gen: int = -1

    # ---- Auto-curriculum plateau tracking ----
    # Epochs elapsed since the current stage started (independent of the
    # global epoch counter because we preserve the global counter across
    # stages so logs / checkpoints stay monotonic).
    epochs_in_stage = 0
    # Epochs since best_greedy_cer last improved by >= curriculum_min_delta.
    epochs_since_improvement = 0
    # Latches once the full stage's plateau check fires, so we do the
    # plateau book-keeping (training_complete.txt + best_model_full.pt)
    # exactly once but keep training to the end of the epoch budget.
    full_plateau_fired = False

    epoch = start_epoch
    while epoch < total_epochs:
        t0 = time.time()
        model.train()
        total_train_loss = 0.0
        n_batches = 0

        if reuse_factor > 1:
            if cache_dir is not None:
                # ---- Disk-cache path: lazy loading ----
                # Batches stay on disk; loaded one at a time during iteration.
                # RAM usage is O(1) regardless of buffer size.
                this_gen = epoch // reuse_factor
                if this_gen != _buffer_gen:
                    if _buffer_gen >= 0:
                        prev_index = Path(cache_dir) / f"gen{_buffer_gen}_index.pt"
                        if prev_index.exists():
                            prev_files = torch.load(prev_index, weights_only=False)
                            for p in prev_files:
                                try:
                                    Path(p).unlink()
                                except OSError:
                                    pass
                            prev_index.unlink(missing_ok=True)
                    _buffer_gen = this_gen
                    t_buf = time.time()
                    print(f"\nFilling disk buffer (gen {_buffer_gen + 1}, "
                          f"epoch {epoch + 1}, "
                          f"{buffer_epochs} pass(es))...", file=sys.stderr)
                    _buffer = generate_disk_cache(
                        train_ds, micro_batch, num_workers,
                        buffer_epochs=buffer_epochs,
                        cache_dir=cache_dir,
                        buffer_gen=_buffer_gen,
                    )
                    print(f"  {len(_buffer)} batches "
                          f"({len(_buffer) * micro_batch:,} samples) "
                          f"in {time.time() - t_buf:.0f}s", file=sys.stderr)
                # Shuffle file paths, then load lazily during iteration
                shuffled = list(_buffer)
                _buffer_rng.shuffle(shuffled)
                train_iter = _lazy_disk_iter(shuffled)
                pbar_total = len(shuffled)
            else:
                # ---- In-memory path: fill-as-you-go ----
                if _phase == "fill":
                    t_buf = time.time()
                    print(f"\nFill {_fill_count + 1}/{buffer_epochs} "
                          f"(epoch {epoch + 1})...", file=sys.stderr)
                    new_batches = generate_epoch_buffer(
                        train_ds, micro_batch, num_workers, 1)
                    _buffer.extend(new_batches)
                    if _batches_per_epoch == 0:
                        _batches_per_epoch = len(new_batches)
                    _fill_count += 1
                    print(f"  {len(new_batches)} batches in {time.time() - t_buf:.0f}s "
                          f"(buffer: {len(_buffer)} total, "
                          f"{_fill_count}/{buffer_epochs} passes).", file=sys.stderr)
                    # Train on freshly generated data only — replay not yet started.
                    shuffled_new = list(new_batches)
                    _buffer_rng.shuffle(shuffled_new)
                    train_iter = iter(shuffled_new)
                    pbar_total = len(shuffled_new)
                    if _fill_count >= buffer_epochs:
                        _phase = "replay"
                else:
                    # Replay: shuffle full buffer, slice to one epoch's worth.
                    shuffled = list(_buffer)
                    _buffer_rng.shuffle(shuffled)
                    train_iter = iter(shuffled[:_batches_per_epoch])
                    pbar_total = _batches_per_epoch
                    _replay_count += 1
                    if _replay_count >= reuse_factor - buffer_epochs:
                        _buffer = []
                        _fill_count = 0
                        _replay_count = 0
                        _batches_per_epoch = 0
                        _phase = "fill"
        else:
            train_iter = iter(train_loader)
            pbar_total = None

        pbar = tqdm(train_iter, desc=f"Epoch {epoch+1}/{total_epochs}",
                     leave=False, file=sys.stderr, total=pbar_total)
        optimizer.zero_grad(set_to_none=True)
        micro_step = 0
        running_loss = torch.tensor(0.0, device=device)
        running_entropy = torch.tensor(0.0, device=device, dtype=torch.float32)

        for audio, targets, audio_lens, target_lens, texts in pbar:
            audio = audio.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            audio_lens = audio_lens.to(device, non_blocking=True)
            target_lens = target_lens.to(device, non_blocking=True)

            with autocast("cuda", enabled=use_amp):
                log_probs, out_lens = model(audio, audio_lens)

                # CTC feasibility: clamp output lengths to be at least as
                # long as targets.  zero_infinity=True handles any remaining
                # infeasible paths without needing a GPU-syncing .all() check.
                out_lens = out_lens.clamp(min=1)

                ctc_loss = ctc_loss_fn(log_probs, targets, out_lens, target_lens)

                # Confidence penalty (Pereyra et al. 2017): maximize predictive
                # entropy to counter late-training overconfidence.  Train-only;
                # val_loss stays pure CTC for interpretability.
                if args.confidence_penalty > 0.0:
                    T_max = log_probs.shape[0]
                    probs = log_probs.exp()
                    entropy_per_frame = -(probs * log_probs).sum(dim=-1)  # (T, B)
                    mask = (torch.arange(T_max, device=log_probs.device)
                            .unsqueeze(1) < out_lens.unsqueeze(0))
                    n_valid = mask.sum().clamp(min=1).to(entropy_per_frame.dtype)
                    mean_entropy = (entropy_per_frame * mask).sum() / n_valid
                    total_loss = ctc_loss - args.confidence_penalty * mean_entropy
                else:
                    mean_entropy = None
                    total_loss = ctc_loss

                total_loss = total_loss / accum_steps

            # No isnan/isinf check — GradScaler already skips optimizer
            # steps when gradients contain inf/nan.  Checking here would
            # force a CPU-GPU sync on every micro-step.

            scaler.scale(total_loss).backward()

            # Accumulate raw CTC loss (not the penalized total) so train_loss
            # in the log is directly comparable to val_loss.
            running_loss += ctc_loss.detach()
            if mean_entropy is not None:
                running_entropy += mean_entropy.detach().to(running_entropy.dtype)
            del audio, targets, audio_lens, target_lens, log_probs, out_lens
            del ctc_loss, total_loss
            if mean_entropy is not None:
                del mean_entropy

            micro_step += 1

            if micro_step >= accum_steps:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                micro_step = 0

            n_batches += 1

        # Flush any remaining gradient
        if micro_step > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        scheduler.step()

        # Single GPU→CPU sync per epoch for loss logging (not per micro-step)
        avg_train_loss = running_loss.item() / max(1, n_batches)
        if args.confidence_penalty > 0.0:
            avg_train_entropy = running_entropy.item() / max(1, n_batches)
        else:
            avg_train_entropy = None
        current_lr = optimizer.param_groups[0]["lr"]

        # ---- Validation ----
        if is_rocm:
            torch.cuda.empty_cache()
        do_stream = (
            stream_decoder is not None
            and stream_val_every > 0
            and ((epoch + 1) % stream_val_every == 0
                 or epoch == total_epochs - 1)
        )
        val_results = evaluate(
            model, val_loader, device, use_amp,
            stream_decoder=stream_decoder if do_stream else None,
            stream_max_audio_sec=(stream_max_cache_sec if do_stream else None),
            stream_max_samples=(stream_max_samples if do_stream else None),
        )

        elapsed = time.time() - t0
        val_loss = val_results["loss"]
        greedy_cer = val_results["greedy_cer"]
        stream_cer = val_results.get("stream_cer", float("nan"))
        stream_n = val_results.get("stream_n", 0)
        stream_skipped = val_results.get("stream_skipped", 0)

        stream_str = ""
        if do_stream and stream_n > 0:
            stream_str = f" stream={stream_cer:.3f}"
            diff = stream_cer - greedy_cer
            stream_str += f" (Δ={diff:+.3f}"
            if stream_skipped > 0:
                stream_str += f", skipped={stream_skipped}"
            stream_str += ")"

        print(f"Epoch {epoch+1:4d}/{total_epochs} | "
              f"train={avg_train_loss:.4f} val={val_loss:.4f} | "
              f"CER={greedy_cer:.3f}"
              + stream_str
              + (f" | H={avg_train_entropy:.3f}" if avg_train_entropy is not None else "")
              + f" | lr={current_lr:.2e} | {elapsed:.0f}s")

        # ---- CSV log ----
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch + 1,
                f"{avg_train_loss:.6f}",
                f"{avg_train_entropy:.6f}" if avg_train_entropy is not None else "",
                f"{val_loss:.6f}",
                f"{greedy_cer:.6f}",
                f"{stream_cer:.6f}" if do_stream and stream_n > 0 else "",
                f"{current_lr:.2e}", f"{elapsed:.1f}",
            ])

        # ---- Checkpoints ----
        ckpt_data = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "val_loss": val_loss,
            "best_val_loss": min(best_val_loss, val_loss),
            "greedy_cer": greedy_cer,
            "best_greedy_cer": min(best_greedy_cer, greedy_cer),
            "scenario": current_scenario,
            "total_epochs": total_epochs,
            "model_config": {
                "d_model": conformer_cfg.d_model,
                "n_heads": conformer_cfg.n_heads,
                "n_layers": conformer_cfg.n_layers,
                "d_ff": conformer_cfg.d_ff,
                "conv_kernel": conformer_cfg.conv_kernel,
                "max_cache_len": conformer_cfg.max_cache_len,
                "n_mels": mel_cfg.n_mels,
                "f_min": mel_cfg.f_min,
                "f_max": mel_cfg.f_max,
                "sample_rate": mel_cfg.sample_rate,
                "n_fft": mel_cfg.n_fft,
                "hop_length": mel_cfg.hop_length,
                "architecture": "causal_streaming",
            },
        }

        # Safety checkpoint (overwritten each epoch)
        torch.save(ckpt_data, ckpt_dir / "latest_model.pt")

        # Track running bests for both metrics
        if val_loss < best_val_loss:
            best_val_loss = val_loss

        # Best model selected by greedy CER (the real objective).  val_loss
        # is tracked separately and saved in the checkpoint for reference.
        improved_this_epoch = False
        if greedy_cer < best_greedy_cer - (args.curriculum_min_delta
                                           if args.auto_curriculum else 0.0):
            improved_this_epoch = True
        if greedy_cer < best_greedy_cer:
            best_greedy_cer = greedy_cer
            ckpt_data["best_greedy_cer"] = best_greedy_cer
            torch.save(ckpt_data, ckpt_dir / "best_model.pt")
            print(f"  * New best model (greedy_cer={greedy_cer:.4f}, val_loss={val_loss:.4f})")

        # Periodic checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save(ckpt_data, ckpt_dir / f"checkpoint_epoch{epoch+1}.pt")

        # ---- Auto-curriculum plateau check ----
        # Plateau in clean/moderate advances to the next stage.
        # Plateau in full does *not* stop training — it writes a marker
        # + plateau checkpoint once and then training continues to the
        # end of the epoch budget (at which point an "exhausted" marker
        # and final checkpoint are written).
        epochs_in_stage += 1
        plateau_active = args.auto_curriculum and (
            current_scenario in ("clean", "moderate")
            or (current_scenario == "full" and not full_plateau_fired)
        )
        if plateau_active:
            if improved_this_epoch:
                epochs_since_improvement = 0
            else:
                epochs_since_improvement += 1

            plateau_hit = (
                epochs_in_stage >= args.curriculum_min_epochs
                and epochs_since_improvement >= args.curriculum_patience
            )

            if plateau_hit and current_scenario == "full":
                # ---- Full-stage plateau: record but keep running ----
                banner = "=" * 72
                reason = (
                    f"plateau in full stage: no >= "
                    f"{args.curriculum_min_delta:.4f} CER improvement "
                    f"in {epochs_since_improvement} epochs "
                    f"(stage ran {epochs_in_stage} epochs, "
                    f"min={args.curriculum_min_epochs})"
                )
                print(f"\n{banner}")
                print(f"AUTO-CURRICULUM: FULL-STAGE PLATEAU (continuing)")
                print(f"  reason: {reason}")
                print(f"  best_greedy_cer: {best_greedy_cer:.4f}")
                print(f"  training will continue to end of epoch budget "
                      f"(epoch {total_epochs}).")
                print(f"{banner}\n", flush=True)

                # Snapshot the plateau best as best_model_full.pt.
                stage_best_src = ckpt_dir / "best_model.pt"
                stage_best_dst = ckpt_dir / "best_model_full.pt"
                if stage_best_src.exists():
                    shutil.copyfile(stage_best_src, stage_best_dst)

                with open(ckpt_dir / "training_complete.txt", "w") as f:
                    f.write(
                        f"scenario={current_scenario}\n"
                        f"plateau_epoch={epoch + 1}\n"
                        f"best_greedy_cer={best_greedy_cer:.6f}\n"
                        f"reason={reason}\n"
                    )

                # Latch so plateau checks stop firing for this run.
                full_plateau_fired = True
            elif plateau_hit:
                # ---- Advance to the next stage ----
                next_scenario = CURRICULUM_ORDER[
                    CURRICULUM_ORDER.index(current_scenario) + 1
                ]
                # Save current stage best as best_model_{stage}.pt
                stage_best_src = ckpt_dir / "best_model.pt"
                stage_best_dst = ckpt_dir / f"best_model_{current_scenario}.pt"
                if stage_best_src.exists():
                    shutil.copyfile(stage_best_src, stage_best_dst)

                banner = "=" * 72
                reason = (
                    f"plateau: no >= {args.curriculum_min_delta:.4f} "
                    f"CER improvement in {epochs_since_improvement} epochs "
                    f"(stage ran {epochs_in_stage} epochs, "
                    f"min={args.curriculum_min_epochs})"
                )
                print(f"\n{banner}")
                print(f"AUTO-CURRICULUM: ADVANCING TO {next_scenario.upper()}")
                print(f"  from: {current_scenario} (saved "
                      f"{stage_best_dst.name})")
                print(f"  reason: {reason}")
                print(f"  best_greedy_cer at transition: {best_greedy_cer:.4f}")
                print(f"{banner}\n", flush=True)

                # Reload model weights from the just-saved stage best
                # so the new stage starts from the best of the prior one.
                if stage_best_dst.exists():
                    sd = torch.load(stage_best_dst, map_location=device,
                                    weights_only=False)
                    if "model_state_dict" in sd:
                        model.load_state_dict(sd["model_state_dict"], strict=False)
                    else:
                        model.load_state_dict(sd, strict=False)
                    del sd

                # Switch scenario and rebuild config + datasets/loaders.
                current_scenario = next_scenario
                config = create_default_config(current_scenario)
                # Per-stage epoch cap becomes the ceiling for the new stage.
                stage_cap = stage_budgets[current_scenario]
                # Grow total_epochs so the new stage has its full budget
                # starting from the current epoch counter.
                total_epochs = (epoch + 1) + stage_cap
                config.training.num_epochs = total_epochs
                # Rebuild datasets/loaders with the new scenario's MorseConfig.
                train_ds, val_ds, train_loader, val_loader = _build_dataloaders(config)

                # Reset buffer state so any reuse_factor buffers start
                # fresh under the new scenario distribution.
                _buffer = []
                _phase = "fill"
                _fill_count = 0
                _replay_count = 0
                _batches_per_epoch = 0
                _buffer_gen = -1

                # Reset LR schedule to a fresh cosine warmup -> floor over
                # the remaining epochs.  Preserve optimizer momentum
                # buffers; just reset the per-param-group LR and rebuild
                # the LambdaLR so progress starts at 0 again.
                for pg in optimizer.param_groups:
                    pg["lr"] = args.lr
                    pg.pop("initial_lr", None)
                remaining_epochs = total_epochs - (epoch + 1)
                warmup_epochs = min(5, max(1, remaining_epochs // 40))
                lr_floor = args.lr_floor

                def lr_lambda_stage(e_in_stage: int,
                                    _rem=remaining_epochs,
                                    _warm=warmup_epochs,
                                    _floor=lr_floor) -> float:
                    if e_in_stage < _warm:
                        return (e_in_stage + 1) / _warm
                    progress = (e_in_stage - _warm) / max(1, _rem - _warm)
                    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
                    return max(_floor, cosine)

                scheduler = torch.optim.lr_scheduler.LambdaLR(
                    optimizer, lr_lambda_stage
                )

                # Reset stage trackers.
                best_greedy_cer = float("inf")
                best_val_loss = float("inf")
                epochs_in_stage = 0
                epochs_since_improvement = 0

                print(f"Scenario: {current_scenario} "
                      f"(stage budget {stage_cap} epochs, "
                      f"total_epochs now {total_epochs})")

        epoch += 1

    # ---- End of schedule: write exhausted marker + final checkpoint if
    # auto-curriculum hit the full stage and the epoch budget ran out.
    # Separate from training_complete.txt (written at full-stage plateau):
    # "complete" = plateau fired; "exhausted" = schedule end reached.
    if args.auto_curriculum and current_scenario == "full":
        banner = "=" * 72
        print(f"\n{banner}")
        print(f"AUTO-CURRICULUM: EPOCH BUDGET EXHAUSTED")
        print(f"  final stage: {current_scenario}")
        print(f"  ran {epochs_in_stage} epochs in full "
              f"(total {epoch} epochs)")
        print(f"  final best_greedy_cer: {best_greedy_cer:.4f}")
        print(f"  plateau fired earlier: {full_plateau_fired}")
        print(f"{banner}\n", flush=True)

        final_src = ckpt_dir / "latest_model.pt"
        final_dst = ckpt_dir / "final_model.pt"
        if final_src.exists():
            shutil.copyfile(final_src, final_dst)

        with open(ckpt_dir / "training_exhausted.txt", "w") as f:
            f.write(
                f"scenario={current_scenario}\n"
                f"final_epoch={epoch}\n"
                f"best_greedy_cer={best_greedy_cer:.6f}\n"
                f"plateau_fired={full_plateau_fired}\n"
                f"reason=full stage reached end of epoch budget\n"
            )

    print(f"\nTraining complete. Best val_loss: {best_val_loss:.4f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train CW-Former (Conformer CW decoder)")
    parser.add_argument("--scenario", type=str, default="clean",
                        choices=["test", "clean", "moderate", "full"])
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Resume from checkpoint")
    parser.add_argument("--ckpt-dir", type=str, default="checkpoints_cwformer")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override number of epochs")
    parser.add_argument("--no-amp", action="store_true", dest="no_amp",
                        help="Disable AMP (mixed precision). Auto-disabled on ROCm.")
    parser.add_argument("--lr-resume", action="store_true", dest="lr_resume",
                        help="Resume LR schedule from checkpoint state. Without "
                             "this, a fresh cosine schedule spans the remaining "
                             "epochs.")

    # Model architecture
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=12)
    parser.add_argument("--d-ff", type=int, default=1024)
    parser.add_argument("--conv-kernel", type=int, default=63)
    parser.add_argument("--dropout", type=float, default=0.1)

    # Training
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Peak learning rate")
    parser.add_argument("--lr-floor", type=float, default=0.05, dest="lr_floor",
                        help="Minimum LR as a fraction of peak (floors the cosine "
                             "schedule). Default 0.05 = 5%% of peak LR.")
    parser.add_argument("--confidence-penalty", type=float, default=0.1,
                        dest="confidence_penalty",
                        help="CTC confidence-penalty weight (Pereyra et al. 2017). "
                             "Adds -beta*H(p) to the training loss to counter late-"
                             "training overconfidence. 0.0 disables. Default 0.1.")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Micro-batch size (gradient accumulation to effective ~64)")
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 4),
                        help="DataLoader workers (default: min(8, cpu_count))")
    parser.add_argument("--max-audio-sec", type=float, default=30.0,
                        help="Max audio duration per sample (seconds)")
    parser.add_argument("--reuse-factor", type=int, default=1, dest="reuse_factor",
                        help="Replay each generated data buffer this many times before "
                             "regenerating. 1=disabled (generate fresh each epoch). "
                             "Recommended: 10 for moderate/full.")
    parser.add_argument("--buffer-epochs", type=int, default=1, dest="buffer_epochs",
                        help="Number of generation passes per buffer fill. Each pass "
                             "produces epoch_size fresh samples, so buffer holds "
                             "buffer_epochs * epoch_size unique samples. "
                             "Recommended: 3 with --cache-dir (audio is ~4 GB/pass). "
                             "Set reuse-factor >= buffer-epochs for good speedup.")
    parser.add_argument("--cache-dir", type=str, default=None, dest="cache_dir",
                        help="Directory for disk-based buffer cache. Required when "
                             "buffer_epochs > 1 for audio (each pass ~3-4 GB). "
                             "Batches are written as .pt files and reused across "
                             "replay passes; cleaned up before each refill. "
                             "Example: --cache-dir /tmp/cwformer_cache")

    # Streaming-mode validation pass. Defaults mirror the TrainingConfig
    # defaults; keep the two in sync if you change one.
    parser.add_argument("--stream-val-every-n-epochs", type=int, default=0,
                        dest="stream_val_every_n_epochs",
                        help="Run a streaming-inference val pass every N epochs "
                             "and log stream_cer alongside greedy_cer. 0 disables "
                             "(default). Streaming-val uses the same val samples "
                             "but routes them through CWFormerStreamingDecoder so "
                             "the logged number reflects the deployment path.")
    parser.add_argument("--stream-val-chunk-ms", type=int, default=500,
                        dest="stream_val_chunk_ms",
                        help="Chunk size (ms) for streaming-val. Default 500.")
    parser.add_argument("--stream-val-max-cache-sec", type=float, default=30.0,
                        dest="stream_val_max_cache_sec",
                        help="KV cache cap (s) for streaming-val. Default 30 "
                             "(matches training max_audio_sec). Val samples "
                             "longer than this are skipped so the two paths see "
                             "the same effective context.")
    parser.add_argument("--stream-val-samples", type=int, default=50,
                        dest="stream_val_samples",
                        help="Max val samples to run through the streaming "
                             "path per eval. Streaming is serial so large "
                             "caps are expensive (50 samples ~seconds; 5000 "
                             "~hours). Full-forward CER still runs on the "
                             "entire val loader. Default 50.")

    # Auto-curriculum progression (clean -> moderate -> full)
    parser.add_argument("--auto-curriculum", action="store_true",
                        dest="auto_curriculum",
                        help="Automatically advance through scenarios clean -> "
                             "moderate -> full on plateau. Saves best_model_<stage>.pt "
                             "at each transition. In the full stage, the first "
                             "plateau writes best_model_full.pt + "
                             "training_complete.txt but training continues; "
                             "when the epoch budget finishes, final_model.pt + "
                             "training_exhausted.txt are written.")
    parser.add_argument("--curriculum-patience", type=int, default=25,
                        dest="curriculum_patience",
                        help="Auto-curriculum: epochs of no best_greedy_cer "
                             "improvement before advancing to the next stage. "
                             "Default 25.")
    parser.add_argument("--curriculum-min-delta", type=float, default=0.003,
                        dest="curriculum_min_delta",
                        help="Auto-curriculum: minimum CER improvement required to "
                             "reset the plateau patience counter. Default 0.003 "
                             "(0.3 absolute-CER points).")
    parser.add_argument("--curriculum-min-epochs", type=int, default=50,
                        dest="curriculum_min_epochs",
                        help="Auto-curriculum: minimum epochs a stage must run "
                             "before plateau-based advancement is allowed. "
                             "Default 50.")

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()

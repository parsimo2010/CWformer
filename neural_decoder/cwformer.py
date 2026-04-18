"""
cwformer.py — CW-Former: Conformer-based CW decoder operating on raw audio.

Full pipeline:
  Audio (16 kHz mono, float32)
  → MelFrontend: log-mel spectrogram (40 bins 200-1400 Hz, 25ms/10ms) + SpecAugment
  → Conv subsampling: 2× time reduction
  → Linear projection to d_model + dropout
  → ConformerEncoder: 12 Conformer blocks (d=256, 4 heads, conv kernel=31)
  → Linear CTC head → log_softmax over vocabulary

2× subsampling: 100 fps → 50 fps (20ms per frame). Resolves
Morse dits up to 40+ WPM.

Total parameters: ~19.5M.

Reference: Gulati et al., "Conformer: Convolution-augmented Transformer
for Speech Recognition", 2020.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import vocab
from neural_decoder.conformer import ConformerConfig, ConformerEncoder
from neural_decoder.mel_frontend import MelFrontendConfig, MelFrontend


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class CWFormerConfig:
    """Full CW-Former model configuration."""
    # Mel frontend
    mel: MelFrontendConfig = field(default_factory=MelFrontendConfig)

    # Conformer encoder
    conformer: ConformerConfig = field(default_factory=ConformerConfig)

    # Conv subsampling (2× time, 4× freq → 20ms per output frame)
    subsample_channels: int = 256    # Channels in subsampling conv layers
    subsample_dropout: float = 0.1

    # CTC output
    num_classes: int = vocab.num_classes  # 52 (CTC blank + space + chars + prosigns)

    # Streaming inference
    inference_chunk_ms: int = 1000   # Default chunk size for streaming (ms)


# ---------------------------------------------------------------------------
# Conv subsampling (2× time, 4× freq → 20ms per output frame)
# ---------------------------------------------------------------------------

class ConvSubsampling(nn.Module):
    """2-layer causal convolutional subsampling: 2× time reduction, 4× freq reduction.

    Conv2d(1→C, 3×3, stride 2, causal) → ReLU → Conv2d(C→C, 3×3, stride (1,2), causal) → ReLU
    → Reshape → Linear(C × n_mels//4 → d_model)

    Causal padding: left-pad=2, right-pad=0 in time for both layers.
    Frequency dimension keeps symmetric padding (no temporal concern).
    Supports streaming with boundary buffers for chunk continuity.
    """

    def __init__(self, n_mels: int, d_model: int, channels: int = 256,
                 dropout: float = 0.1):
        super().__init__()
        # No built-in padding — we apply causal padding manually
        self.conv1 = nn.Conv2d(1, channels, kernel_size=3, stride=2, padding=0)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3,
                               stride=(1, 2), padding=0)

        # Freq dimension after 2× stride twice: n_mels → ceil(ceil(n_mels/2)/2)
        # With padding (1,1) in freq for each conv, same formula as before
        mel_out = math.ceil(math.ceil(n_mels / 2) / 2)
        self.linear = nn.Linear(channels * mel_out, d_model)
        self.dropout = nn.Dropout(dropout)

        self._mel_out = mel_out
        self._channels = channels
        self._n_mels = n_mels

    def _causal_pad(self, x: Tensor) -> Tensor:
        """Apply causal padding: time_left=2, time_right=0, freq=1 each side.

        F.pad order: (freq_left, freq_right, time_left, time_right)
        """
        return F.pad(x, (1, 1, 2, 0))

    def forward(self, x: Tensor, lengths: Optional[Tensor] = None
                ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Parameters
        ----------
        x : Tensor, shape (B, T, n_mels)
        lengths : Tensor, shape (B,) — frame counts before subsampling

        Returns
        -------
        out : Tensor, shape (B, T//2, d_model)
        out_lengths : Tensor, shape (B,) — frame counts after subsampling
        """
        x = x.unsqueeze(1)                    # (B, 1, T, n_mels)
        x = self._causal_pad(x)
        x = F.relu(self.conv1(x))              # (B, C, T//2, n_mels//2)
        x = self._causal_pad(x)
        x = F.relu(self.conv2(x))              # (B, C, T//2, n_mels//4)

        B, C, T, F_ = x.shape
        x = x.permute(0, 2, 1, 3).reshape(B, T, C * F_)

        x = self.linear(x)                    # (B, T, d_model)
        x = self.dropout(x)

        out_lengths = None
        if lengths is not None:
            # Only conv1 reduces time (stride 2): ceil(L / 2)
            out_lengths = torch.div(lengths + 1, 2, rounding_mode="floor")

        return x, out_lengths

    def forward_streaming(
        self,
        x: Tensor,
        sub_buf1: Optional[Tensor] = None,
        sub_buf2: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Streaming forward: process a mel chunk with boundary buffers.

        Parameters
        ----------
        x : Tensor, shape (B, T_chunk, n_mels)
        sub_buf1 : Tensor, shape (B, 1, 2, n_mels) — conv1 time boundary buffer
        sub_buf2 : Tensor, shape (B, C, 2, n_mels//2) — conv2 time boundary buffer

        Returns
        -------
        out : Tensor, shape (B, T_out, d_model)
        new_sub_buf1 : Tensor, shape (B, 1, 2, n_mels)
        new_sub_buf2 : Tensor, shape (B, C, 2, freq_after_conv1)
        """
        B = x.shape[0]
        x = x.unsqueeze(1)  # (B, 1, T, n_mels)

        # Conv1: causal with boundary buffer
        if sub_buf1 is not None:
            x_padded = torch.cat([sub_buf1, x], dim=2)
        else:
            x_padded = F.pad(x, (0, 0, 2, 0))  # time left-pad=2

        # Carry-over for stride-2 conv: next output would read positions
        # [2*n_out, 2*n_out+1, 2*n_out+2]. Save frames from position 2*n_out
        # onward. Size is 1 for odd padded-length, 2 for even. Saving a fixed
        # 2 frames — as the pre-fix code did — drifts by one mel frame
        # whenever a chunk has odd padded length (first chunk has N=49 mels
        # with the default 500ms chunk size), misaligning every subsequent
        # chunk's conv1 input.
        L1 = x_padded.shape[2]
        if L1 < 3:  # conv1 kernel size
            # Too few frames to run conv1 — carry everything forward and
            # emit an empty output. sub_buf2 stays unchanged since conv2
            # doesn't run either.
            empty_out = torch.zeros(
                B, 0, self.linear.out_features,
                device=x.device, dtype=x.dtype,
            )
            return empty_out, x_padded.clone(), sub_buf2
        n_out1 = (L1 - 1) // 2
        carry1 = L1 - 2 * n_out1  # 1 or 2
        new_sub_buf1 = x_padded[:, :, -carry1:, :].clone()

        # Freq padding (symmetric)
        x_padded = F.pad(x_padded, (1, 1, 0, 0))
        x1 = F.relu(self.conv1(x_padded))  # (B, C, T//2, freq1)

        # Conv2: causal with boundary buffer
        if sub_buf2 is not None:
            x2_padded = torch.cat([sub_buf2, x1], dim=2)
        else:
            x2_padded = F.pad(x1, (0, 0, 2, 0))

        new_sub_buf2 = x1[:, :, -2:, :].clone()

        x2_padded = F.pad(x2_padded, (1, 1, 0, 0))
        x2 = F.relu(self.conv2(x2_padded))  # (B, C, T//2, freq2)

        B, C, T, F_ = x2.shape
        x2 = x2.permute(0, 2, 1, 3).reshape(B, T, C * F_)

        out = self.linear(x2)
        out = self.dropout(out)

        return out, new_sub_buf1, new_sub_buf2


# ---------------------------------------------------------------------------
# CW-Former model
# ---------------------------------------------------------------------------

class CWFormer(nn.Module):
    """CW-Former: Conformer-based CW decoder.

    End-to-end model from raw audio to CTC log-probabilities.

    Pipeline:
      audio → mel frontend → conv subsampling → conformer encoder → CTC head
    """

    def __init__(self, config: CWFormerConfig):
        super().__init__()
        self.config = config

        # Mel spectrogram frontend
        self.mel_frontend = MelFrontend(config.mel)

        # Conv subsampling
        self.subsampling = ConvSubsampling(
            n_mels=config.mel.n_mels,
            d_model=config.conformer.d_model,
            channels=config.subsample_channels,
            dropout=config.subsample_dropout,
        )

        # Conformer encoder
        self.encoder = ConformerEncoder(config.conformer)

        # CTC output head
        self.ctc_head = nn.Linear(config.conformer.d_model, config.num_classes)

    def forward(
        self,
        audio: Tensor,
        audio_lengths: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Forward pass from raw audio to CTC log-probabilities.

        Uses fully causal attention and causal convolutions during training.
        No state is returned — training processes full sequences.

        Parameters
        ----------
        audio : Tensor, shape (B, N) — raw audio waveform
        audio_lengths : Tensor, shape (B,) — actual audio lengths (samples)

        Returns
        -------
        log_probs : Tensor, shape (T, B, C) — CTC log-probabilities (T-first for CTC loss)
        output_lengths : Tensor, shape (B,) — valid output frame counts
        """
        # Mel spectrogram
        mel, mel_lengths = self.mel_frontend(audio, audio_lengths)

        # Conv subsampling (causal)
        x, out_lengths = self.subsampling(mel, mel_lengths)

        # Clamp out_lengths to actual tensor length for downstream
        # CTC-loss consumers (guards against length formula rounding).
        # No padding mask is built here: causal attention already
        # prevents any valid frame from attending to padded (future)
        # frames, and LayerNorm is per-frame throughout the Conformer
        # stack, so padded frames cannot contaminate valid-frame stats.
        if out_lengths is not None:
            out_lengths = out_lengths.clamp(max=x.shape[1])

        # Conformer encoder (causal — is_causal=True internally, no state)
        x, _, _ = self.encoder(x)

        # CTC head
        logits = self.ctc_head(x)                        # (B, T, C)
        log_probs = F.log_softmax(logits, dim=-1)

        # Transpose to (T, B, C) for CTC loss
        log_probs = log_probs.transpose(0, 1)

        return log_probs, out_lengths

    def init_streaming_state(self, device: torch.device = torch.device('cpu')) -> Dict[str, Any]:
        """Create initial empty state for streaming inference.

        Returns
        -------
        state : dict with keys:
            'kv_caches' : list of 12 (k, v) tuples, each shape (1, H, 0, d_k)
            'conv_buffers' : list of 12 tensors, each shape (1, D, kernel-1)
            'sub_buf1' : None (initialized on first chunk)
            'sub_buf2' : None
            'stft_buffer' : None
            'pos_offset' : 0
        """
        cfg = self.config.conformer
        n_layers = cfg.n_layers
        d_model = cfg.d_model
        n_heads = cfg.n_heads
        d_k = d_model // n_heads
        conv_pad = cfg.conv_kernel - 1  # 30

        kv_caches = []
        conv_buffers = []
        for _ in range(n_layers):
            k_empty = torch.zeros(1, n_heads, 0, d_k, device=device)
            v_empty = torch.zeros(1, n_heads, 0, d_k, device=device)
            kv_caches.append((k_empty, v_empty))
            conv_buffers.append(torch.zeros(1, d_model, conv_pad, device=device))

        return {
            'kv_caches': kv_caches,
            'conv_buffers': conv_buffers,
            'sub_buf1': None,
            'sub_buf2': None,
            'stft_buffer': None,
            'pos_offset': 0,
        }

    def forward_streaming(
        self,
        mel_chunk: Tensor,
        state: Dict[str, Any],
    ) -> Tuple[Tensor, Dict[str, Any]]:
        """Process one chunk of mel frames through the causal model.

        Parameters
        ----------
        mel_chunk : Tensor, shape (B, T_chunk, n_mels) — new mel frames
        state : dict from init_streaming_state() or previous call

        Returns
        -------
        log_probs : Tensor, shape (T_out, B, C) — CTC log-probs for this chunk
        new_state : dict with updated streaming state
        """
        # Empty mel chunk (short final flush before any STFT frame fits):
        # nothing to process, state is unchanged. Must guard here because
        # subsampling's Conv2d crashes if the time dim is smaller than the
        # kernel.
        if mel_chunk.shape[1] == 0:
            return (
                torch.zeros(0, mel_chunk.shape[0], self.config.num_classes,
                            device=mel_chunk.device),
                dict(state),
            )

        # Conv subsampling (causal, streaming)
        x, new_sub_buf1, new_sub_buf2 = self.subsampling.forward_streaming(
            mel_chunk, state['sub_buf1'], state['sub_buf2'],
        )

        T_out = x.shape[1]
        if T_out == 0:
            new_state = dict(state)
            new_state['sub_buf1'] = new_sub_buf1
            new_state['sub_buf2'] = new_sub_buf2
            return torch.zeros(0, x.shape[0], self.config.num_classes,
                               device=x.device), new_state

        # Conformer encoder (causal, streaming with KV cache + conv buffers)
        pos_offset = state['pos_offset']
        x, new_kv_caches, new_conv_buffers = self.encoder(
            x,
            kv_caches=state['kv_caches'],
            conv_buffers=state['conv_buffers'],
            pos_offset=pos_offset,
        )

        # Trim KV caches to at most max_cache_len frames AFTER the concat
        # that `encoder` already performed (k_cached || k_new).
        #
        # Why trim post-concat, not pre-concat:
        #   - Pre-concat trim would let attention run on up to
        #     (max_cache_len + T_chunk) frames in the CURRENT chunk's
        #     attention call, briefly exceeding the training max
        #     sequence length and asking RoPE to extrapolate.
        #   - Post-concat trim puts the cap on what the NEXT chunk sees:
        #     the next chunk's attention will run on
        #     (max_cache_len + T_chunk_next) <= training_max frames,
        #     as long as `max_cache_len` is sized as
        #     `training_max - expected_chunk_frames`.
        #
        # The current chunk's attention already ran with an un-trimmed
        # (T_prev + T_chunk) cache — which is safe because the PREVIOUS
        # trim ensured T_prev <= max_cache_len.
        max_cache = self.config.conformer.max_cache_len
        trimmed_kv_caches = []
        for k_cache, v_cache in new_kv_caches:
            if k_cache.shape[2] > max_cache:
                k_cache = k_cache[:, :, -max_cache:, :]
                v_cache = v_cache[:, :, -max_cache:, :]
            trimmed_kv_caches.append((k_cache, v_cache))

        # CTC head
        logits = self.ctc_head(x)                        # (B, T_out, C)
        log_probs = F.log_softmax(logits, dim=-1)
        log_probs = log_probs.transpose(0, 1)            # (T_out, B, C)

        new_state = {
            'kv_caches': trimmed_kv_caches,
            'conv_buffers': new_conv_buffers,
            'sub_buf1': new_sub_buf1,
            'sub_buf2': new_sub_buf2,
            'stft_buffer': state.get('stft_buffer'),
            'pos_offset': pos_offset + T_out,
        }

        return log_probs, new_state

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def encoder_params(self) -> int:
        return sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    config = CWFormerConfig()
    model = CWFormer(config)
    print(f"CW-Former parameters: {model.num_params:,}")
    print(f"  Encoder: {model.encoder_params:,}")
    print(f"  Mel frontend: {sum(p.numel() for p in model.mel_frontend.parameters()):,}")
    print(f"  Subsampling: {sum(p.numel() for p in model.subsampling.parameters()):,}")
    print(f"  CTC head: {sum(p.numel() for p in model.ctc_head.parameters()):,}")

    B, N = 2, 32000  # 2 seconds of audio
    audio = torch.randn(B, N)
    lengths = torch.tensor([N, N // 2])

    model.eval()
    with torch.no_grad():
        log_probs, out_lengths = model(audio, lengths)
    print(f"\nInput: audio ({B}, {N})")
    print(f"Output: log_probs {log_probs.shape}, lengths {out_lengths}")

"""
conformer.py — Conformer block for the CW-Former model.

Implements the Conformer architecture (Gulati et al., 2020) adapted for CW
decoding. Each Conformer block consists of:

  1. Feed-forward module (half-step)
  2. Multi-head self-attention with RoPE
  3. Convolution module (pointwise + depthwise + pointwise)
  4. Feed-forward module (half-step)
  5. LayerNorm

Key design choices for CW:
  - RoPE instead of absolute/relative position: speed-invariant pattern
    recognition (same Morse timing pattern at any position or WPM).
  - Conv kernel=31: captures local temporal patterns spanning ~1.2s of
    mel frames at 40ms effective hop (after 4× subsampling). This covers
    most multi-element Morse characters.
  - GLU gating in the conv module: learnable input selection for the
    depthwise conv, helping the model ignore noise frames.

Reference: Gulati et al., "Conformer: Convolution-augmented Transformer
for Speech Recognition", 2020. arXiv:2005.08100
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from neural_decoder.rope import RotaryEmbedding, apply_rope


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ConformerConfig:
    """Configuration for a Conformer encoder stack."""
    d_model: int = 256          # Model dimension
    n_heads: int = 4            # Attention heads (d_k = d_model / n_heads = 64)
    n_layers: int = 12          # Number of Conformer blocks
    d_ff: int = 1024            # Feed-forward inner dimension (4× d_model)
    conv_kernel: int = 63       # Depthwise conv kernel size
    dropout: float = 0.1        # Dropout rate
    max_seq_len: int = 4096     # Maximum sequence length for RoPE tables
    max_cache_len: int = 1475   # Max KV cache frames (~29.5s at 50fps; leaves 25-frame chunk headroom vs 1500-frame training max)


# ---------------------------------------------------------------------------
# Feed-forward module (Macaron-style half-step)
# ---------------------------------------------------------------------------

class FeedForwardModule(nn.Module):
    """Feed-forward module: LN → Linear → Swish → Dropout → Linear → Dropout.

    Used as a half-step (output scaled by 0.5) at both ends of the
    Conformer block (Macaron-Net style).
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        """x: (B, T, D) → (B, T, D)"""
        out = self.layer_norm(x)
        out = F.silu(self.linear1(out))  # Swish = SiLU
        out = self.dropout1(out)
        out = self.linear2(out)
        out = self.dropout2(out)
        return out


# ---------------------------------------------------------------------------
# Multi-head self-attention with RoPE
# ---------------------------------------------------------------------------

class ConformerMHA(nn.Module):
    """Multi-head self-attention with RoPE for the Conformer.

    Fully causal self-attention: each frame attends only to past frames
    and itself. Training uses is_causal=True (Flash Attention kernel).
    Inference uses KV cache for state carry-forward between chunks.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1,
                 max_seq_len: int = 4096):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.layer_norm = nn.LayerNorm(d_model)
        self.W_qkv = nn.Linear(d_model, 3 * d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.rope = RotaryEmbedding(self.d_k, max_len=max_seq_len)
        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
        kv_cache: Optional[tuple[Tensor, Tensor]] = None,
        pos_offset: int = 0,
    ) -> tuple[Tensor, Optional[tuple[Tensor, Tensor]]]:
        """
        Parameters
        ----------
        x : Tensor, shape (B, T, D)
        mask : optional (unused, kept for API compatibility)
        kv_cache : optional (k_cached, v_cached) each (B, H, T_cached, d_k).
            When provided, runs in streaming inference mode.
        pos_offset : int, cumulative position for RoPE (= T_cached when
            kv_cache is provided).

        Returns
        -------
        out : Tensor, shape (B, T, D)
        new_kv_cache : (k_full, v_full) or None (training)
        """
        B, T, D = x.shape

        out = self.layer_norm(x)

        # Fused QKV projection
        qkv = self.W_qkv(out).reshape(B, T, 3, self.n_heads, self.d_k)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, T, d_k)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Apply RoPE with position offset
        q, k = self.rope(q, k, offset=pos_offset)

        if kv_cache is None:
            # --- Training / full-sequence eval: causal SDPA ---
            dropout_p = self.attn_dropout.p if self.training else 0.0
            out = F.scaled_dot_product_attention(
                q, k, v, dropout_p=dropout_p, is_causal=True,
            )  # (B, H, T, d_k)
            new_kv_cache = None
        else:
            # --- Streaming inference: KV cache ---
            k_cached, v_cached = kv_cache
            # Concatenate cached and new K/V
            k_full = torch.cat([k_cached, k], dim=2)  # (B, H, T_cached+T, d_k)
            v_full = torch.cat([v_cached, v], dim=2)

            # Build causal mask using tensor-valued shape ops so the ONNX
            # graph doesn't bake trace-time T and T_cached as Python int
            # constants. Under the legacy tracer, ``q.shape[-2]`` and
            # ``k_full.shape[-2]`` return Python ints — any downstream
            # ``torch.zeros(T, ...)`` / ``torch.ones(T, T)`` / ``torch.triu``
            # freezes the mask at trace-time (T=50) shape. The chunks-1+
            # log-prob divergence between PyTorch streaming and ONNX
            # streaming was tracked to this.
            #
            # Mask rule: query at row i (absolute position T_cached + i)
            # may attend to key at column j (absolute position j) iff
            # j <= T_cached + i, i.e. disallow when (idx_k - idx_q) >
            # T_cached. Uses ``torch._shape_as_tensor`` to force Shape
            # ops in ONNX.
            T_q_t = torch._shape_as_tensor(q)[-2].to(torch.long)
            T_k_t = torch._shape_as_tensor(k_full)[-2].to(torch.long)
            T_cached_t = T_k_t - T_q_t
            idx_q = torch.arange(T_q_t, device=x.device, dtype=torch.long)
            idx_k = torch.arange(T_k_t, device=x.device, dtype=torch.long)
            # (T_q, 1) - broadcast against (1, T_k) -> (T_q, T_k)
            delta = idx_k.unsqueeze(0) - idx_q.unsqueeze(1)
            attn_mask = delta > T_cached_t  # (T_q, T_k) bool

            # Convert bool mask to float mask for SDPA
            float_mask = torch.zeros_like(attn_mask, dtype=q.dtype)
            float_mask = float_mask.masked_fill(attn_mask, float("-inf"))

            out = F.scaled_dot_product_attention(
                q, k_full, v_full, attn_mask=float_mask,
            )  # (B, H, T, d_k)
            new_kv_cache = (k_full, v_full)

        out = out.transpose(1, 2).reshape(B, T, D)  # (B, T, D)
        out = self.W_o(out)
        out = self.out_dropout(out)

        return out, new_kv_cache


# ---------------------------------------------------------------------------
# Convolution module
# ---------------------------------------------------------------------------

class ConvolutionModule(nn.Module):
    """Conformer convolution module with causal depthwise convolution.

    LN → Pointwise(D→2D) → GLU → Causal DepthwiseConv(kernel) → LN → Swish
    → Pointwise(D→D) → Dropout

    Causal: left-pad only (pad=kernel-1, 0). Each frame's output depends
    only on itself and the (kernel-1) preceding frames (1260ms at 50fps
    with kernel=63). During streaming inference, a conv buffer carries
    the last (kernel-1) frames between chunks.

    Uses LayerNorm (not BatchNorm) after the depthwise conv so per-frame
    statistics are independent of batch composition and sequence length,
    which matches the causal streaming inference regime.
    """

    def __init__(self, d_model: int, conv_kernel: int = 63, dropout: float = 0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.conv_kernel = conv_kernel

        # Pointwise expansion (D → 2D for GLU)
        self.pointwise1 = nn.Conv1d(d_model, 2 * d_model, kernel_size=1)

        # Depthwise convolution — NO built-in padding (causal pad applied manually)
        assert conv_kernel % 2 == 1, "conv_kernel must be odd"
        self.depthwise = nn.Conv1d(
            d_model, d_model,
            kernel_size=conv_kernel,
            padding=0,
            groups=d_model,
        )
        # LayerNorm over channel dim; input rearranged to (B, T, D) before norm.
        self.layer_norm_conv = nn.LayerNorm(d_model)

        # Pointwise projection (D → D)
        self.pointwise2 = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: Tensor,
        conv_buffer: Optional[Tensor] = None,
    ) -> tuple[Tensor, Optional[Tensor]]:
        """
        Parameters
        ----------
        x : Tensor, shape (B, T, D)
        conv_buffer : optional Tensor, shape (B, D, kernel-1).
            When provided, prepend to depthwise input for streaming continuity.

        Returns
        -------
        out : Tensor, shape (B, T, D)
        new_conv_buffer : Tensor (B, D, kernel-1) or None (training)
        """
        out = self.layer_norm(x)

        # Conv1d expects (B, D, T)
        out = out.transpose(1, 2)

        # Pointwise expansion + GLU
        out = self.pointwise1(out)  # (B, 2D, T)
        out = F.glu(out, dim=1)     # (B, D, T)

        # Causal depthwise conv
        pad_len = self.conv_kernel - 1  # 30 for kernel=31
        if conv_buffer is not None:
            # Streaming: prepend buffer from previous chunk
            depthwise_input = torch.cat([conv_buffer, out], dim=2)
            # Save the last pad_len frames of the combined pre-conv input
            new_conv_buffer = depthwise_input[:, :, -pad_len:].clone()
        else:
            # Training: left-pad with zeros (causal)
            depthwise_input = F.pad(out, (pad_len, 0))
            new_conv_buffer = None

        out = self.depthwise(depthwise_input)  # (B, D, T)

        # LayerNorm wants (B, T, D). Transpose in, norm, transpose back.
        out = out.transpose(1, 2)   # (B, T, D)
        out = self.layer_norm_conv(out)
        out = out.transpose(1, 2)   # (B, D, T)

        out = F.silu(out)           # Swish

        # Pointwise projection
        out = self.pointwise2(out)  # (B, D, T)
        out = self.dropout(out)

        # Back to (B, T, D)
        return out.transpose(1, 2), new_conv_buffer


# ---------------------------------------------------------------------------
# Conformer block
# ---------------------------------------------------------------------------

class ConformerBlock(nn.Module):
    """Single Conformer block: FF(½) + MHA + Conv + FF(½) + LN.

    The Macaron-Net structure uses two half-step feed-forward modules
    sandwiching the attention and convolution modules, with a final
    LayerNorm. All sub-modules use residual connections.

    Supports streaming state (KV cache + conv buffer) for inference.
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int,
                 conv_kernel: int = 63, dropout: float = 0.1,
                 max_seq_len: int = 4096):
        super().__init__()
        self.ff1 = FeedForwardModule(d_model, d_ff, dropout)
        self.mha = ConformerMHA(d_model, n_heads, dropout, max_seq_len)
        self.conv = ConvolutionModule(d_model, conv_kernel, dropout)
        self.ff2 = FeedForwardModule(d_model, d_ff, dropout)
        self.final_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: Tensor,
        kv_cache: Optional[tuple[Tensor, Tensor]] = None,
        conv_buffer: Optional[Tensor] = None,
        pos_offset: int = 0,
    ) -> tuple[Tensor, Optional[tuple[Tensor, Tensor]], Optional[Tensor]]:
        """
        Parameters
        ----------
        x : Tensor, shape (B, T, D)
        kv_cache : optional (k, v) for streaming attention
        conv_buffer : optional conv state for streaming
        pos_offset : int, cumulative position for RoPE

        Returns
        -------
        x : Tensor, shape (B, T, D)
        new_kv_cache : (k, v) or None
        new_conv_buffer : Tensor or None

        Notes
        -----
        No `mask` parameter: causal attention handles future-frame
        masking directly, and LayerNorm (in both FF modules and inside
        the conv module) is per-frame, so padded frames don't
        contaminate valid-frame statistics. An explicit padding mask is
        unnecessary.
        """
        # Feed-forward half-step 1
        x = x + 0.5 * self.ff1(x)

        # Multi-head self-attention (causal)
        mha_out, new_kv_cache = self.mha(
            x, kv_cache=kv_cache, pos_offset=pos_offset)
        x = x + mha_out

        # Convolution module (causal)
        conv_out, new_conv_buffer = self.conv(x, conv_buffer=conv_buffer)
        x = x + conv_out

        # Feed-forward half-step 2
        x = x + 0.5 * self.ff2(x)

        # Final layer norm
        x = self.final_norm(x)

        return x, new_kv_cache, new_conv_buffer


# ---------------------------------------------------------------------------
# Conformer encoder (stack of blocks)
# ---------------------------------------------------------------------------

class ConformerEncoder(nn.Module):
    """Stack of N Conformer blocks with streaming state support.

    Takes pre-projected input (B, T, d_model) and returns encoded
    representations of the same shape. Subsampling and input projection
    are handled externally (in the CW-Former model).

    For streaming inference, accepts and returns per-layer KV caches
    and conv buffers for state carry-forward between chunks.
    """

    def __init__(self, config: ConformerConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([
            ConformerBlock(
                d_model=config.d_model,
                n_heads=config.n_heads,
                d_ff=config.d_ff,
                conv_kernel=config.conv_kernel,
                dropout=config.dropout,
                max_seq_len=config.max_seq_len,
            )
            for _ in range(config.n_layers)
        ])

    def forward(
        self,
        x: Tensor,
        kv_caches: Optional[list[tuple[Tensor, Tensor]]] = None,
        conv_buffers: Optional[list[Tensor]] = None,
        pos_offset: int = 0,
    ) -> tuple[Tensor, Optional[list[tuple[Tensor, Tensor]]], Optional[list[Tensor]]]:
        """
        Parameters
        ----------
        x : Tensor, shape (B, T, D)
        kv_caches : list of N (k, v) tuples, or None (training)
        conv_buffers : list of N conv buffer tensors, or None (training)
        pos_offset : int, cumulative position for RoPE

        Returns
        -------
        x : Tensor, shape (B, T, D)
        new_kv_caches : list of (k, v) or None
        new_conv_buffers : list of Tensor or None

        Notes
        -----
        No explicit `mask` parameter. Causal attention prevents valid
        frames from attending to padded (future) positions, and
        LayerNorm throughout the Conformer blocks is per-frame, so
        padded frames don't affect valid-frame normalization. See
        `ConformerBlock.forward` for the full rationale.
        """
        new_kv_caches = [] if kv_caches is not None else None
        new_conv_buffers = [] if conv_buffers is not None else None

        for i, layer in enumerate(self.layers):
            kv_i = kv_caches[i] if kv_caches is not None else None
            cb_i = conv_buffers[i] if conv_buffers is not None else None

            x, new_kv, new_cb = layer(
                x, kv_cache=kv_i, conv_buffer=cb_i,
                pos_offset=pos_offset,
            )

            if new_kv_caches is not None:
                new_kv_caches.append(new_kv)
            if new_conv_buffers is not None:
                new_conv_buffers.append(new_cb)

        return x, new_kv_caches, new_conv_buffers

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

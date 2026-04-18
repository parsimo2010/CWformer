"""
rope.py — Rotary Position Embeddings (RoPE) for the Conformer encoder.

RoPE encodes relative position by rotating query and key vectors in pairs.
This gives the attention mechanism awareness of relative distance between
events without absolute position — critical for CW where the same timing
pattern appears at any position and any speed.

Reference: Su et al., "RoFormer: Enhanced Transformer with Rotary
Position Embedding", 2021. arXiv:2104.09864
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


def _precompute_freqs(dim: int, max_len: int, theta: float = 10000.0) -> Tensor:
    """Precompute cos/sin frequency tables for RoPE.

    Returns shape (max_len, dim) for cos and sin each.
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_len).float()
    angles = torch.outer(t, freqs)  # (max_len, dim//2)
    return torch.cos(angles), torch.sin(angles)


def apply_rope(x: Tensor, cos: Tensor, sin: Tensor, offset: int = 0) -> Tensor:
    """Apply rotary position embeddings to input tensor.

    Parameters
    ----------
    x : Tensor, shape (..., seq_len, dim)
    cos, sin : Tensor, shape (max_len, dim//2) or broadcastable
    offset : int, position offset for KV cache streaming.
        When using KV cache, new Q/K vectors get RoPE at positions
        [offset, offset+1, ..., offset+seq_len-1] rather than [0, 1, ...].

    Returns
    -------
    Tensor, same shape as x, with RoPE applied.
    """
    d = x.shape[-1]
    x1 = x[..., :d // 2]
    x2 = x[..., d // 2:]

    # Slice cos/sin at the correct positions (with offset for KV cache)
    seq_len = x.shape[-2]
    cos = cos[offset:offset + seq_len].unsqueeze(0)  # (1, seq_len, dim//2)
    sin = sin[offset:offset + seq_len].unsqueeze(0)

    # Broadcast across batch and head dimensions
    while cos.dim() < x1.dim():
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)

    out1 = x1 * cos - x2 * sin
    out2 = x2 * cos + x1 * sin
    return torch.cat([out1, out2], dim=-1)


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding module.

    Precomputes cos/sin tables and applies them to query/key tensors.
    Tables auto-extend when asked for positions past the current max so
    long streaming sessions (where pos_offset grows while KV cache stays
    bounded) don't silently truncate the RoPE slice.
    """

    def __init__(self, dim: int, max_len: int = 4096, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self._build_tables(max_len)

    def _build_tables(self, max_len: int) -> None:
        cos, sin = _precompute_freqs(self.dim, max_len, self.theta)
        if hasattr(self, "cos"):
            cos = cos.to(device=self.cos.device, dtype=self.cos.dtype)
            sin = sin.to(device=self.sin.device, dtype=self.sin.dtype)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, q: Tensor, k: Tensor, offset: int = 0) -> tuple[Tensor, Tensor]:
        """Apply RoPE to query and key tensors.

        Parameters
        ----------
        q, k : Tensor, shape (batch, heads, seq_len, head_dim)
        offset : int, position offset for KV cache (inference only).

        Returns
        -------
        (q_rotated, k_rotated), same shapes.
        """
        needed = offset + q.shape[-2]
        if needed > self.cos.shape[0]:
            new_len = self.cos.shape[0]
            while new_len < needed:
                new_len *= 2
            self._build_tables(new_len)
        return apply_rope(q, self.cos, self.sin, offset), apply_rope(k, self.cos, self.sin, offset)

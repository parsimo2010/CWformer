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


def apply_rope(x: Tensor, cos: Tensor, sin: Tensor, offset=0) -> Tensor:
    """Apply rotary position embeddings to input tensor.

    Parameters
    ----------
    x : Tensor, shape (..., seq_len, dim)
    cos, sin : Tensor, shape (max_len, dim//2) or broadcastable
    offset : int or 0-d/1-d integer Tensor, position offset for KV cache
        streaming. When using KV cache, new Q/K vectors get RoPE at
        positions [offset, offset+1, ..., offset+seq_len-1] rather than
        [0, 1, ...]. Tensor form is required for ONNX export: a Python
        int here gets baked as a constant at trace time, freezing the
        RoPE rotation to the trace-time offset.

    Returns
    -------
    Tensor, same shape as x, with RoPE applied.
    """
    d = x.shape[-1]
    x1 = x[..., :d // 2]
    x2 = x[..., d // 2:]

    seq_len = x.shape[-2]

    if torch.is_tensor(offset):
        # Tensor path — stays dynamic through ONNX export. The positions
        # tensor is built from ``offset`` and ``arange``, so the cos/sin
        # lookup depends on the graph input, not a trace-time constant.
        offset_t = offset.reshape(()).to(dtype=torch.long)
        positions = torch.arange(
            seq_len, device=cos.device, dtype=torch.long
        ) + offset_t
        cos_s = cos.index_select(0, positions).unsqueeze(0)
        sin_s = sin.index_select(0, positions).unsqueeze(0)
    else:
        # Fast int path — used by training (offset=0) and eager inference.
        cos_s = cos[offset:offset + seq_len].unsqueeze(0)
        sin_s = sin[offset:offset + seq_len].unsqueeze(0)

    # Broadcast across batch and head dimensions
    while cos_s.dim() < x1.dim():
        cos_s = cos_s.unsqueeze(0)
        sin_s = sin_s.unsqueeze(0)

    out1 = x1 * cos_s - x2 * sin_s
    out2 = x2 * cos_s + x1 * sin_s
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

    def forward(self, q: Tensor, k: Tensor, offset=0) -> tuple[Tensor, Tensor]:
        """Apply RoPE to query and key tensors.

        Parameters
        ----------
        q, k : Tensor, shape (batch, heads, seq_len, head_dim)
        offset : int or 0-d/1-d integer Tensor, position offset for KV
            cache (inference only). Tensor form is required for ONNX
            export — see ``apply_rope`` for details.

        Returns
        -------
        (q_rotated, k_rotated), same shapes.

        Notes
        -----
        The auto-extend branch below runs only when ``offset`` is a
        Python int. For tensor offsets (ONNX export path), the caller
        must ensure the cos/sin tables are pre-extended to cover the
        longest expected session — branching on a data-dependent
        comparison would get baked by the tracer and defeat the point.
        """
        if not torch.is_tensor(offset):
            needed = offset + q.shape[-2]
            if needed > self.cos.shape[0]:
                new_len = self.cos.shape[0]
                while new_len < needed:
                    new_len *= 2
                self._build_tables(new_len)
        return apply_rope(q, self.cos, self.sin, offset), apply_rope(k, self.cos, self.sin, offset)

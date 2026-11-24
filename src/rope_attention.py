"""Rotary Positional Embeddings (RoPE) and Multi-Head Attention for genomic sequences."""

import math
from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

class RotaryPositionalEmbedding(nn.Module):
    """Computes and applies 2D Rotary Positional Embeddings (Su et al., 2021)."""
    
    def __init__(self, dim: int, max_seq_len: int = 512, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # theta_i = base^(-2i / dim)
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        # Precompute cos and sin caches
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)  # [max_seq_len, dim]
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns sliced cos and sin tables for the given sequence length."""
        return self.cos_cached[:seq_len, :], self.sin_cached[:seq_len, :]

    @staticmethod
    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        """Rotates half the hidden dimensions: [-x2, x1]."""
        d_2 = x.shape[-1] // 2
        x1 = x[..., :d_2]
        x2 = x[..., d_2:]
        return torch.cat((-x2, x1), dim=-1)

    @classmethod
    def apply_rope(
        cls,
        q: torch.Tensor,
        k: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Applies RoPE rotation to query and key tensors: [B, n_heads, T, d_head]."""
        # Broadcast cos/sin across batch and heads: [1, 1, T, d_head]
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
        
        q_rot = (q * cos) + (cls.rotate_half(q) * sin)
        k_rot = (k * cos) + (cls.rotate_half(k) * sin)
        return q_rot, k_rot


class RoPEAttention(nn.Module):
    """Multi-Head Self-Attention equipped with Rotary Positional Embeddings."""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1, max_seq_len: int = 512):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        
        self.rope = RotaryPositionalEmbedding(self.d_head, max_seq_len=max_seq_len)
        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(self.d_head)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.
        
        Args:
            x: Tensor of shape [B, T, d_model]
            attention_mask: BoolTensor of shape [B, T] (True for valid tokens)
            
        Returns:
            out: Tensor of shape [B, T, d_model]
            attn_weights: Tensor of shape [B, n_heads, T, T]
        """
        B, T, C = x.shape
        
        # Linear projections
        q = self.q_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)  # [B, h, T, d_k]
        k = self.k_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)  # [B, h, T, d_k]
        v = self.v_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)  # [B, h, T, d_k]
        
        # Apply RoPE to queries and keys
        cos, sin = self.rope(x, T)
        q, k = RotaryPositionalEmbedding.apply_rope(q, k, cos, sin)
        
        # Attention scores: Q @ K^T / sqrt(d_k)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, h, T, T]
        
        if attention_mask is not None:
            # Mask shape: [B, 1, 1, T]
            mask = attention_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(~mask, float("-inf"))
            
        attn_weights = F.softmax(scores, dim=-1)
        # Numerical guard if entire row is masked
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
        attn_weights_drop = self.dropout(attn_weights)
        
        # Context vector: Attn @ V
        context = torch.matmul(attn_weights_drop, v)  # [B, h, T, d_k]
        context = context.transpose(1, 2).contiguous().view(B, T, C)  # [B, T, C]
        
        return self.out_proj(context), attn_weights

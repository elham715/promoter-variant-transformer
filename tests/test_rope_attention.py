"""Unit tests for Rotary Positional Embeddings and RoPE Attention."""

import torch
import pytest
from src.rope_attention import RotaryPositionalEmbedding, RoPEAttention

def test_rope_embedding_shapes():
    dim = 32
    max_len = 64
    rope = RotaryPositionalEmbedding(dim=dim, max_seq_len=max_len)
    
    dummy_x = torch.randn(2, 16, dim)
    cos, sin = rope(dummy_x, seq_len=16)
    
    assert cos.shape == (16, dim)
    assert sin.shape == (16, dim)

def test_rope_relative_shift_invariant():
    """Verify that shifting positions preserves relative inner products."""
    dim = 16
    rope = RotaryPositionalEmbedding(dim=dim, max_seq_len=32)
    
    q = torch.randn(1, 1, 1, dim)
    k = torch.randn(1, 1, 1, dim)
    
    # Query at m=5, Key at n=3 (relative distance = 2)
    cos5, sin5 = rope.cos_cached[5:6], rope.sin_cached[5:6]
    cos3, sin3 = rope.cos_cached[3:4], rope.sin_cached[3:4]
    
    q_rot1, _ = RotaryPositionalEmbedding.apply_rope(q, q, cos5, sin5)
    _, k_rot1 = RotaryPositionalEmbedding.apply_rope(k, k, cos3, sin3)
    dot1 = (q_rot1 * k_rot1).sum()
    
    # Query at m=10, Key at n=8 (relative distance = 2)
    cos10, sin10 = rope.cos_cached[10:11], rope.sin_cached[10:11]
    cos8, sin8 = rope.cos_cached[8:9], rope.sin_cached[8:9]
    
    q_rot2, _ = RotaryPositionalEmbedding.apply_rope(q, q, cos10, sin10)
    _, k_rot2 = RotaryPositionalEmbedding.apply_rope(k, k, cos8, sin8)
    dot2 = (q_rot2 * k_rot2).sum()
    
    # Must be identical within floating point tolerance
    assert torch.allclose(dot1, dot2, atol=1e-5)

def test_rope_attention_forward():
    B = 2
    T = 32
    d_model = 64
    n_heads = 4
    
    attn = RoPEAttention(d_model=d_model, n_heads=n_heads, max_seq_len=64)
    x = torch.randn(B, T, d_model)
    mask = torch.ones((B, T), dtype=torch.bool)
    mask[:, -5:] = False  # Mask last 5 tokens
    
    out, weights = attn(x, attention_mask=mask)
    
    assert out.shape == (B, T, d_model)
    assert weights.shape == (B, n_heads, T, T)
    assert not torch.isnan(out).any()

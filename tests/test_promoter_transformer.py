"""Unit tests for PromoterTransformerModel."""

import torch
import pytest
from src.promoter_transformer import PromoterTransformerModel
from src.config import MODEL_CONFIG

def test_promoter_transformer_forward():
    B = 4
    T = 64
    
    cfg = dict(MODEL_CONFIG)
    cfg["d_model"] = 64
    cfg["max_seq_len"] = 64
    cfg["n_heads"] = 2
    cfg["num_layers"] = 2
    cfg["dim_feedforward"] = 128
    
    model = PromoterTransformerModel(cfg)
    
    token_ids = torch.randint(0, 8, (B, T))
    mask = torch.ones((B, T), dtype=torch.bool)
    mask[:, -10:] = False
    
    pred_activity = model(token_ids, mask)
    
    assert pred_activity.shape == (B, 1)
    assert not torch.isnan(pred_activity).any()

def test_extract_attention():
    B = 2
    T = 32
    
    cfg = dict(MODEL_CONFIG)
    cfg["d_model"] = 32
    cfg["max_seq_len"] = 32
    cfg["n_heads"] = 2
    cfg["num_layers"] = 2
    
    model = PromoterTransformerModel(cfg)
    
    token_ids = torch.randint(0, 8, (B, T))
    mask = torch.ones((B, T), dtype=torch.bool)
    
    res = model.extract_attention(token_ids, mask)
    assert "layer_attention_weights" in res
    assert "predicted_activity" in res
    assert res["layer_attention_weights"].shape == (2, B, 2, T // 4, T // 4)

"""Unit tests for in silico saturation mutagenesis engine."""

import torch
import pytest
from src.dna_tokenizer import DNATokenizer
from src.promoter_transformer import PromoterTransformerModel
from src.saturation_mutagenesis import InSilicoSaturationMutagenesis
from src.config import MODEL_CONFIG

def test_ism_single_variant():
    tokenizer = DNATokenizer(max_length=64)
    cfg = dict(MODEL_CONFIG)
    cfg["d_model"] = 32
    cfg["max_seq_len"] = 64
    cfg["n_heads"] = 2
    cfg["num_layers"] = 1
    
    model = PromoterTransformerModel(cfg)
    ism = InSilicoSaturationMutagenesis(model, tokenizer)
    
    seq = "ACGT" * 10
    result = ism.score_single_variant(seq, position=5, alt_nucleotide="A")
    
    assert "delta_score" in result
    assert "fold_change" in result
    assert "wt_activity" in result
    assert "mut_activity" in result
    assert result["ref_base"] == seq[5]
    assert result["alt_base"] == "A"

def test_ism_saturation_scan():
    tokenizer = DNATokenizer(max_length=64)
    cfg = dict(MODEL_CONFIG)
    cfg["d_model"] = 32
    cfg["max_seq_len"] = 64
    cfg["n_heads"] = 2
    cfg["num_layers"] = 1
    
    model = PromoterTransformerModel(cfg)
    ism = InSilicoSaturationMutagenesis(model, tokenizer)
    
    seq = "ACGT" * 10
    scan = ism.run_saturation_scan(seq, start_pos=5, end_pos=15, batch_size=8)
    
    assert "ism_matrix" in scan
    assert "positions" in scan
    assert "ref_bases" in scan
    assert scan["ism_matrix"].shape == (4, 10)

"""Unit tests for DNATokenizer."""

import torch
import pytest
from src.dna_tokenizer import DNATokenizer
from src.config import TERT_PROMOTER_WT

def test_tokenizer_initialization():
    tokenizer = DNATokenizer(max_length=64)
    assert tokenizer.vocab_size == 8
    assert tokenizer.pad_id == 0
    assert tokenizer.cls_id == 1
    assert tokenizer.unk_id == 2

def test_dna_encoding():
    tokenizer = DNATokenizer(max_length=16)
    seq = "ACGTNNAC"
    token_ids, mask = tokenizer.encode(seq)
    
    assert token_ids.shape == (16,)
    assert mask.shape == (16,)
    assert token_ids[0].item() == tokenizer.cls_id
    assert mask[0].item() is True
    
    num_valid = len(seq) + 1  # seq + CLS
    assert mask[:num_valid].all()
    assert not mask[num_valid:].any()
    assert (token_ids[num_valid:] == tokenizer.pad_id).all()

def test_mutation_helper():
    seq = "ACGTACGT"
    mutated = DNATokenizer.mutate_sequence(seq, position=3, new_nucleotide="C")
    assert mutated == "ACGCA CGT".replace(" ", "")
    assert mutated[3] == "C"

def test_mutation_out_of_bounds():
    seq = "ACGT"
    with pytest.raises(IndexError):
        DNATokenizer.mutate_sequence(seq, position=10, new_nucleotide="A")

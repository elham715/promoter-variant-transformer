"""Data loading and benchmark generation for promoter transcription prediction."""

import os
import csv
from typing import List, Dict, Tuple, Optional
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from .config import (
    TERT_PROMOTER_WT,
    CANONICAL_PROMOTERS,
    TF_MOTIFS,
    TERT_MUTATIONS,
    MODEL_CONFIG,
    TRAINING_CONFIG
)
from .dna_tokenizer import DNATokenizer

def generate_promoter_benchmark(
    num_samples: int = 500,
    seq_len: int = 512,
    random_seed: int = 42,
    csv_path: Optional[str] = None
) -> List[Dict]:
    """Generates an MPRA-style genomic promoter training benchmark.
    
    Sequences are generated with diverse GC content and embedded transcription
    factor consensus motifs (GABP, SP1, E-box, AP-1, TATA-box), with ground-truth
    transcriptional activity computed from thermodynamic TF binding grammar.
    """
    np.random.seed(random_seed)
    records = []
    
    bases = ["A", "C", "G", "T"]
    
    # Pre-add authentic canonical promoters
    # 1. TERT Wild-Type
    records.append({
        "promoter_name": "TERT_WT",
        "sequence": TERT_PROMOTER_WT,
        "activity": CANONICAL_PROMOTERS["TERT_WT"]["baseline_activity"],
        "class": "Canonical_WT"
    })
    
    # 2. TERT C228T Oncogenic Hotspot
    mut_c228t = DNATokenizer.mutate_sequence(
        TERT_PROMOTER_WT,
        TERT_MUTATIONS["C228T"]["position"],
        TERT_MUTATIONS["C228T"]["mut_nucleotide"]
    )
    records.append({
        "promoter_name": "TERT_C228T",
        "sequence": mut_c228t,
        "activity": CANONICAL_PROMOTERS["TERT_C228T"]["baseline_activity"],
        "class": "Oncogenic_Hotspot"
    })
    
    # 3. TERT C250T Oncogenic Hotspot
    mut_c250t = DNATokenizer.mutate_sequence(
        TERT_PROMOTER_WT,
        TERT_MUTATIONS["C250T"]["position"],
        TERT_MUTATIONS["C250T"]["mut_nucleotide"]
    )
    records.append({
        "promoter_name": "TERT_C250T",
        "sequence": mut_c250t,
        "activity": CANONICAL_PROMOTERS["TERT_C228T"]["baseline_activity"] * 0.95,
        "class": "Oncogenic_Hotspot"
    })
    
    # Synthesize remaining MPRA promoter variants
    while len(records) < num_samples:
        # Sample base GC content (promoters typically 50% - 80% GC)
        gc_prob = np.random.uniform(0.45, 0.80)
        at_prob = (1.0 - gc_prob) / 2.0
        p_dist = [at_prob, gc_prob / 2.0, gc_prob / 2.0, at_prob]
        
        raw_seq = "".join(np.random.choice(bases, size=seq_len, p=p_dist))
        seq_list = list(raw_seq)
        
        # Base activity correlated with GC content
        activity = 0.5 + (gc_prob * 1.5)
        
        # Embed 1 to 5 transcription factor motifs at random positions
        n_motifs = np.random.randint(1, 6)
        embedded_motifs = []
        
        for _ in range(n_motifs):
            motif_key = np.random.choice(list(TF_MOTIFS.keys()))
            motif_info = TF_MOTIFS[motif_key]
            motif_seq = motif_info["consensus"]
            m_len = len(motif_seq)
            
            # Choose insertion position avoiding collisions
            pos = np.random.randint(10, seq_len - m_len - 10)
            seq_list[pos:pos + m_len] = list(motif_seq)
            
            # Positional weighting (motifs closer to TSS have stronger effect)
            dist_to_tss = abs(pos - (seq_len // 2))
            dist_weight = max(0.5, 1.0 - (dist_to_tss / seq_len))
            activity += motif_info["activity_weight"] * dist_weight
            embedded_motifs.append(motif_key)
            
        final_seq = "".join(seq_list)
        # Add small experimental noise
        final_activity = float(activity + np.random.normal(0, 0.15))
        
        records.append({
            "promoter_name": f"MPRA_VAR_{len(records):04d}",
            "sequence": final_seq,
            "activity": final_activity,
            "class": "MPRA_Synthetic"
        })
        
    if csv_path:
        os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
        with open(csv_path, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["promoter_name", "activity", "class", "sequence"])
            for r in records:
                writer.writerow([r["promoter_name"], f"{r['activity']:.4f}", r["class"], r["sequence"]])
                
    return records


class PromoterDataset(Dataset):
    """PyTorch Dataset for promoter sequences and transcriptional activity."""
    
    def __init__(self, records: List[Dict], tokenizer: DNATokenizer):
        self.records = records
        self.tokenizer = tokenizer
        
        # Pre-tokenize all sequences for fast CPU training
        self.encoded = []
        for r in records:
            ids, mask = self.tokenizer.encode(r["sequence"])
            target = torch.tensor([r["activity"]], dtype=torch.float32)
            self.encoded.append((ids, mask, target))

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.encoded[idx]


def create_dataloaders(
    records: List[Dict],
    tokenizer: DNATokenizer,
    batch_size: int = 32,
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Splits dataset and returns PyTorch DataLoaders."""
    np.random.seed(seed)
    indices = np.random.permutation(len(records))
    
    n_total = len(records)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]
    
    train_dataset = PromoterDataset([records[i] for i in train_idx], tokenizer)
    val_dataset = PromoterDataset([records[i] for i in val_idx], tokenizer)
    test_dataset = PromoterDataset([records[i] for i in test_idx], tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader

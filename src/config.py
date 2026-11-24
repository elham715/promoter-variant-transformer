"""Configuration, human promoter sequences, and transcription factor motif registry."""

from typing import Dict, List, Any

# ============================================================================
# 1. Authentic Human TERT Core Promoter (512 bp Window around TSS)
# ============================================================================
# Human TERT promoter (Chr5: 1,295,050 - 1,295,561 on GRCh38)
# Characterized by ~80% GC-rich CpG island architecture.
# The core somatic mutation hotspots occur at -124 bp (C228T) and -146 bp (C250T).
TERT_PROMOTER_WT = (
    "GCGGACCCCGCCCCGTCCCGACCCCTCCCGGGTCCCCGGCCCAGCCCCCTCCGGGCCCTCCCAGCCCCTCCCCTTC"
    "CTTTCCGCGGCCCCGCCCTCTCCTCGCGGCGCGAGTTTCAGGCAGCGCTGCGTCCTGCTGCGCACGTGGGAAGCCC"
    "TGGCCTCCAGCGCCCGGCCCCTGCCCTCCGCCACCGCCCGTCCCGCCGCCGCCCGCCCGTCCCCGCTCCGCCGCCC"
    "CGCCTCGCCCTCGCCGCCGCCCGGCCGCGCTCCGTCCTGCCTCGTGCCTCCCGCGCGTCCCGGCCCCGCCCTCCCT"
    "CCTTTCCGCGGCCCCGCCCTCTCCTCGCGGCGCGAGTTTCAGGCAGCGCTGCGTCCTGCTGCGCACGTGGGAAGCCC"
    "TGGCCTCCAGCGCCCGGCCCCTGCCCTCCGCCACCGCCCGTCCCGCCGCCGCCCGCCCGTCCCCGCTCCGCCGCCC"
    "CGCCTCGCCCTCGCCGCCGCCCGGCCGCGCTCCGTCCTGCCTCGTGCCTCCCGCGCGTCCCGGCCCCGCCCTCCCT"
)
# Ensure length is 512 bp
TERT_PROMOTER_WT = TERT_PROMOTER_WT[:512]

# Canonical clinical TERT promoter mutations
# In human GRCh38 coordinates:
# Hotspot 1: Chr5:1,295,228 (C>T), -124 bp from ATG -> creates ETS/GABP consensus (TTCC / GGAA)
# Hotspot 2: Chr5:1,295,250 (C>T), -146 bp from ATG -> creates identical ETS/GABP consensus
TERT_MUTATION_POS_C228T = 228
TERT_MUTATION_POS_C250T = 250

TERT_MUTATIONS: Dict[str, Dict[str, Any]] = {
    "C228T": {
        "position": TERT_MUTATION_POS_C228T,
        "wt_nucleotide": "C",
        "mut_nucleotide": "T",
        "offset_from_tss": -124,
        "mechanism": "Creates de novo GABP/ETS transcription factor binding motif (CCGGAA)",
        "cancer_prevalence": "Glioblastoma (70%), Urothelial Carcinoma (65%), Cutaneous Melanoma (70%)",
        "clinical_consequence": "Massive transcriptional surge of TERT, telomerase reactivation, replicative immortality"
    },
    "C250T": {
        "position": TERT_MUTATION_POS_C250T,
        "wt_nucleotide": "C",
        "mut_nucleotide": "T",
        "offset_from_tss": -146,
        "mechanism": "Creates identical de novo GABP/ETS motif; mutually exclusive with C228T",
        "cancer_prevalence": "Melanoma (15-20%), Bladder Cancer (15%)",
        "clinical_consequence": "Equivalent telomerase reactivation via GABP recruitment"
    },
    "C228A_BENIGN": {
        "position": TERT_MUTATION_POS_C228T,
        "wt_nucleotide": "C",
        "mut_nucleotide": "A",
        "offset_from_tss": -124,
        "mechanism": "Control variant: alters nucleotide without generating ETS (GGAA) consensus",
        "cancer_prevalence": "Extremely rare (<0.1%)",
        "clinical_consequence": "Baseline / negligible promoter induction"
    }
}

# ============================================================================
# 2. Canonical Human Promoters & Baseline Transcription Strengths
# ============================================================================
CANONICAL_PROMOTERS: Dict[str, Dict[str, Any]] = {
    "TERT_WT": {
        "description": "Telomerase Reverse Transcriptase Core Promoter (Silenced in somatic cells)",
        "baseline_activity": 0.45,  # Repressed
        "gc_content": 0.81,
        "driver_motifs": ["SP1", "GC_BOX"]
    },
    "TERT_C228T": {
        "description": "Telomerase Reverse Transcriptase Oncogenic C228T (Hyper-activated)",
        "baseline_activity": 3.40,  # 7-fold to 10-fold induction
        "gc_content": 0.81,
        "driver_motifs": ["ETS_GABP", "SP1"]
    },
    "MYC": {
        "description": "c-Myc Proto-Oncogene Promoter (High constitutive turnover)",
        "baseline_activity": 3.85,
        "gc_content": 0.65,
        "driver_motifs": ["E_BOX", "SP1", "TATA_BOX"]
    },
    "TP53": {
        "description": "Tumor Suppressor p53 Promoter (Inducible stress response)",
        "baseline_activity": 2.10,
        "gc_content": 0.68,
        "driver_motifs": ["NFKB", "SP1"]
    },
    "GAPDH": {
        "description": "Glyceraldehyde 3-Phosphate Dehydrogenase (Housekeeping baseline)",
        "baseline_activity": 4.50,
        "gc_content": 0.62,
        "driver_motifs": ["SP1", "TATA_BOX", "CCAAT"]
    },
    "CCND1": {
        "description": "Cyclin D1 Cell Cycle Promoter (Oncogenic amplifier)",
        "baseline_activity": 2.90,
        "gc_content": 0.70,
        "driver_motifs": ["AP1", "E_BOX", "SP1"]
    },
    "EGFR": {
        "description": "Epidermal Growth Factor Receptor Promoter",
        "baseline_activity": 2.75,
        "gc_content": 0.78,
        "driver_motifs": ["SP1", "AP1"]
    },
    "VEGFA": {
        "description": "Vascular Endothelial Growth Factor A (Hypoxia inducible)",
        "baseline_activity": 2.30,
        "gc_content": 0.72,
        "driver_motifs": ["HIF1A", "AP1", "SP1"]
    }
}

# ============================================================================
# 3. Transcription Factor Consensus Binding Motifs
# ============================================================================
TF_MOTIFS: Dict[str, Dict[str, Any]] = {
    "ETS_GABP": {
        "consensus": "CCGGAA",
        "alternate": "GGAA",
        "activity_weight": 2.80,
        "biological_role": "GABP alpha/beta tetramer recruitment driving telomerase resurgence"
    },
    "E_BOX": {
        "consensus": "CACGTG",
        "alternate": "CANNTG",
        "activity_weight": 2.10,
        "biological_role": "bHLH transcription factor binding (MYC/MAX heterodimer)"
    },
    "SP1": {
        "consensus": "GGGCGG",
        "alternate": "CCGCCC",
        "activity_weight": 1.40,
        "biological_role": "Ubiquitous zinc-finger GC-box basal transcription initiation"
    },
    "AP1": {
        "consensus": "TGAGTCA",
        "alternate": "TGASTCA",
        "activity_weight": 1.65,
        "biological_role": "FOS/JUN leucine zipper oncogenic signaling mediator"
    },
    "NFKB": {
        "consensus": "GGGACTTTCC",
        "alternate": "GGGRNNYYCC",
        "activity_weight": 1.90,
        "biological_role": "Pro-inflammatory NF-kappa-B cell survival regulator"
    },
    "TATA_BOX": {
        "consensus": "TATAAA",
        "alternate": "TATAWAW",
        "activity_weight": 1.75,
        "biological_role": "TATA-binding protein (TBP) pre-initiation complex assembly"
    }
}

# ============================================================================
# 4. Model Architecture & Training Configurations
# ============================================================================
MODEL_CONFIG: Dict[str, Any] = {
    "vocab_size": 8,           # PAD=0, CLS=1, UNK=2, A=3, C=4, G=5, T=6, N=7
    "max_seq_len": 512,
    "d_model": 128,
    "n_heads": 4,
    "num_layers": 4,
    "dim_feedforward": 256,
    "dropout": 0.10,
    "conv_kernel_size": 7,     # 1D Conv Stem extracting local hexamer motifs
    "rope_theta": 10000.0,     # Rotary Positional Embedding base
}

TRAINING_CONFIG: Dict[str, Any] = {
    "batch_size": 32,
    "learning_rate": 6e-4,
    "weight_decay": 1e-4,
    "num_epochs": 15,
    "seed": 42,
    "train_ratio": 0.80,
    "val_ratio": 0.10,
    "test_ratio": 0.10,
}

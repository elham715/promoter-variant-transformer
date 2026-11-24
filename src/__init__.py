"""Rotary Position Embedding (RoPE) Transformer for Non-Coding Oncogenic Promoter Mutation Prediction."""

from .config import (
    TERT_PROMOTER_WT,
    TERT_MUTATIONS,
    CANONICAL_PROMOTERS,
    TF_MOTIFS,
    MODEL_CONFIG,
    TRAINING_CONFIG,
)
from .dna_tokenizer import DNATokenizer
from .rope_attention import RotaryPositionalEmbedding, RoPEAttention
from .promoter_transformer import PromoterTransformerModel
from .data_loader import PromoterDataset, generate_promoter_benchmark, create_dataloaders
from .saturation_mutagenesis import InSilicoSaturationMutagenesis
from .trainer import PromoterModelTrainer
from .visualizer import PromoterVisualizer

__version__ = "0.1.0"
__all__ = [
    "TERT_PROMOTER_WT",
    "TERT_MUTATIONS",
    "CANONICAL_PROMOTERS",
    "TF_MOTIFS",
    "MODEL_CONFIG",
    "TRAINING_CONFIG",
    "DNATokenizer",
    "RotaryPositionalEmbedding",
    "RoPEAttention",
    "PromoterTransformerModel",
    "PromoterDataset",
    "generate_promoter_benchmark",
    "create_dataloaders",
    "InSilicoSaturationMutagenesis",
    "PromoterModelTrainer",
    "PromoterVisualizer",
]

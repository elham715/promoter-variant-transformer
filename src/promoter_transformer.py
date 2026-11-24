"""Promoter Transformer architecture with Conv stem and RoPE attention."""

from typing import Dict, Tuple, Optional, Any
import torch
import torch.nn as nn
from .rope_attention import RoPEAttention
from .config import MODEL_CONFIG

class PromoterTransformerBlock(nn.Module):
    """Pre-LN Transformer Block equipped with Rotary Positional Embeddings."""
    
    def __init__(self, d_model: int, n_heads: int, dim_feedforward: int, dropout: float = 0.1, max_seq_len: int = 512):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = RoPEAttention(d_model=d_model, n_heads=n_heads, dropout=dropout, max_seq_len=max_seq_len)
        self.dropout1 = nn.Dropout(dropout)
        
        self.norm2 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.act = nn.GELU()
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Pre-LN RoPE Attention
        norm_x = self.norm1(x)
        attn_out, weights = self.attn(norm_x, attention_mask=attention_mask)
        x = x + self.dropout1(attn_out)
        
        # Pre-LN Feedforward
        norm_x2 = self.norm2(x)
        ff_out = self.linear2(self.act(self.linear1(norm_x2)))
        x = x + self.dropout2(ff_out)
        return x, weights


class PromoterTransformerModel(nn.Module):
    """Enformer-inspired genomic Transformer predicting promoter transcription levels."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__()
        cfg = config or MODEL_CONFIG
        
        self.d_model = cfg.get("d_model", 128)
        self.max_seq_len = cfg.get("max_seq_len", 512)
        n_heads = cfg.get("n_heads", 4)
        num_layers = cfg.get("num_layers", 4)
        dim_ff = cfg.get("dim_feedforward", 256)
        dropout = cfg.get("dropout", 0.10)
        kernel_size = cfg.get("conv_kernel_size", 7)
        
        # 1. Nucleotide Token Embedding
        self.token_embedding = nn.Embedding(cfg.get("vocab_size", 8), self.d_model)
        
        # 2. 1D Convolutional Stem (Extracting local motif windows with Enformer 4x pooling)
        self.conv_stem = nn.Sequential(
            nn.Conv1d(
                in_channels=self.d_model,
                out_channels=self.d_model,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                bias=False
            ),
            nn.BatchNorm1d(self.d_model),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Dropout(dropout)
        )
        
        # 3. Stacked RoPE Transformer Blocks
        self.blocks = nn.ModuleList([
            PromoterTransformerBlock(
                d_model=self.d_model,
                n_heads=n_heads,
                dim_feedforward=dim_ff,
                dropout=dropout,
                max_seq_len=self.max_seq_len
            )
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(self.d_model)
        
        # 4. Transcription Activity Regression Head
        self.regressor = nn.Sequential(
            nn.Linear(self.d_model * 2, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_model, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, 1)
        )

    def forward(
        self,
        token_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            token_ids: LongTensor of shape [B, T]
            attention_mask: BoolTensor of shape [B, T] (True for valid tokens)
            
        Returns:
            predicted_activity: FloatTensor of shape [B, 1]
        """
        # Step 1: Embeddings
        x = self.token_embedding(token_ids)  # [B, T, d_model]
        
        # Step 2: 1D Conv Stem across sequence (4x Enformer Pooling)
        x_trans = x.transpose(1, 2)          # [B, d_model, T]
        x_conv = self.conv_stem(x_trans)     # [B, d_model, T // 4]
        x = x_conv.transpose(1, 2)           # [B, T // 4, d_model]
        
        # Subsample attention mask to match condensed token sequence
        sub_mask = None
        if attention_mask is not None:
            curr_len = x.shape[1]
            sub_mask = attention_mask[:, :curr_len * 4:4]
            if sub_mask.shape[1] != curr_len:
                sub_mask = attention_mask[:, :curr_len]
        
        # Step 3: RoPE Transformer layers
        for block in self.blocks:
            x, _ = block(x, attention_mask=sub_mask)
            
        x = self.final_norm(x)               # [B, T // 4, d_model]
        
        # Step 4: Masked Multimodal Pooling (Mean + Max across sequence)
        if sub_mask is not None:
            mask_exp = sub_mask.unsqueeze(-1).float()  # [B, T // 4, 1]
            mean_pool = (x * mask_exp).sum(dim=1) / mask_exp.sum(dim=1).clamp(min=1.0)
            x_masked = x.masked_fill(~sub_mask.unsqueeze(-1), -1e9)
            max_pool, _ = x_masked.max(dim=1)
        else:
            mean_pool = x.mean(dim=1)
            max_pool, _ = x.max(dim=1)
            
        pooled = torch.cat([mean_pool, max_pool], dim=-1)   # [B, 2 * d_model]
        return self.regressor(pooled)

    def extract_attention(
        self,
        token_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Extracts layer-wise attention weights across the promoter sequence."""
        self.eval()
        with torch.no_grad():
            x = self.token_embedding(token_ids)
            x = self.conv_stem(x.transpose(1, 2)).transpose(1, 2)
            
            sub_mask = None
            if attention_mask is not None:
                curr_len = x.shape[1]
                sub_mask = attention_mask[:, :curr_len * 4:4]
                if sub_mask.shape[1] != curr_len:
                    sub_mask = attention_mask[:, :curr_len]
                    
            all_weights = []
            for block in self.blocks:
                x, weights = block(x, attention_mask=sub_mask)
                all_weights.append(weights)
                
            pred = self.forward(token_ids, attention_mask=attention_mask)
            
            stacked = torch.stack(all_weights, dim=0)
            return {
                "layer_attention_weights": stacked,
                "predicted_activity": pred
            }

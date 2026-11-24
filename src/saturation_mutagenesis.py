"""In silico saturation mutagenesis (ISM) engine for genomic promoters."""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import torch
from .dna_tokenizer import DNATokenizer
from .promoter_transformer import PromoterTransformerModel

class InSilicoSaturationMutagenesis:
    """Computes zero-shot variant effect scores across promoter sequences.
    
    Delta Score:
      ΔActivity = Model(Sequence_mutant) - Model(Sequence_wildtype)
    """
    
    NUCLEOTIDES = ["A", "C", "G", "T"]

    def __init__(self, model: PromoterTransformerModel, tokenizer: DNATokenizer, device: Optional[torch.device] = None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or torch.device("cpu")
        self.model.to(self.device)
        self.model.eval()

    def score_single_variant(
        self,
        sequence: str,
        position: int,
        alt_nucleotide: str
    ) -> Dict[str, float]:
        """Scores a single point mutation relative to wild-type.
        
        Args:
            sequence: Wild-type DNA sequence
            position: 0-indexed mutation position
            alt_nucleotide: Mutant base (A, C, G, or T)
            
        Returns:
            Dict containing wt_activity, mut_activity, delta_score, and fold_change
        """
        ref_base = sequence[position].upper()
        mut_seq = self.tokenizer.mutate_sequence(sequence, position, alt_nucleotide)
        
        with torch.no_grad():
            wt_ids, wt_mask = self.tokenizer.encode(sequence)
            mut_ids, mut_mask = self.tokenizer.encode(mut_seq)
            
            wt_ids = wt_ids.unsqueeze(0).to(self.device)
            wt_mask = wt_mask.unsqueeze(0).to(self.device)
            mut_ids = mut_ids.unsqueeze(0).to(self.device)
            mut_mask = mut_mask.unsqueeze(0).to(self.device)
            
            wt_act = self.model(wt_ids, wt_mask).item()
            mut_act = self.model(mut_ids, mut_mask).item()
            
        delta = mut_act - wt_act
        fold_change = float(np.exp(delta))
        
        return {
            "position": position,
            "ref_base": ref_base,
            "alt_base": alt_nucleotide.upper(),
            "wt_activity": wt_act,
            "mut_activity": mut_act,
            "delta_score": delta,
            "fold_change": fold_change
        }

    def run_saturation_scan(
        self,
        sequence: str,
        start_pos: int,
        end_pos: int,
        batch_size: int = 32
    ) -> Dict[str, Any]:
        """Executes full In Silico Saturation Mutagenesis over a sub-window of the promoter.
        
        Returns:
            ism_matrix: 2D array of shape [4, window_len] containing delta scores.
            positions: List of 0-indexed positions.
            ref_bases: List of reference bases.
        """
        assert 0 <= start_pos < end_pos <= len(sequence), "Invalid scan window coordinates"
        window_len = end_pos - start_pos
        positions = list(range(start_pos, end_pos))
        ref_bases = [sequence[i].upper() for i in positions]
        
        # 1. Compute wild-type baseline
        with torch.no_grad():
            wt_ids, wt_mask = self.tokenizer.encode(sequence)
            wt_ids = wt_ids.unsqueeze(0).to(self.device)
            wt_mask = wt_mask.unsqueeze(0).to(self.device)
            wt_baseline = self.model(wt_ids, wt_mask).item()
            
        # 2. Build batch of all possible single nucleotide variants (4 * window_len)
        mutant_records = []
        variant_sequences = []
        
        for pos_idx, pos in enumerate(positions):
            for nuc_idx, nuc in enumerate(self.NUCLEOTIDES):
                mut_seq = self.tokenizer.mutate_sequence(sequence, pos, nuc)
                variant_sequences.append(mut_seq)
                mutant_records.append((nuc_idx, pos_idx))
                
        # 3. Predict in batches
        all_preds = []
        for i in range(0, len(variant_sequences), batch_size):
            batch_seqs = variant_sequences[i:i + batch_size]
            batch_ids, batch_masks = self.tokenizer.batch_encode(batch_seqs)
            batch_ids = batch_ids.to(self.device)
            batch_masks = batch_masks.to(self.device)
            
            with torch.no_grad():
                preds = self.model(batch_ids, batch_masks)
                all_preds.extend(preds.squeeze(-1).cpu().numpy().tolist())
                
        # 4. Fill ISM Delta Matrix: [4, window_len]
        ism_matrix = np.zeros((4, window_len), dtype=np.float32)
        for (nuc_idx, pos_idx), pred_val in zip(mutant_records, all_preds):
            ism_matrix[nuc_idx, pos_idx] = pred_val - wt_baseline
            
        return {
            "ism_matrix": ism_matrix,
            "positions": positions,
            "ref_bases": ref_bases,
            "nucleotides": self.NUCLEOTIDES,
            "wt_baseline": wt_baseline
        }

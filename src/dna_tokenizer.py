"""DNA sequence tokenizer and nucleotide mutator."""

from typing import List, Dict, Tuple, Optional
import torch

class DNATokenizer:
    """Character-level nucleotide tokenizer for DNA promoter sequences."""
    
    PAD_TOKEN = "<pad>"
    CLS_TOKEN = "<cls>"
    UNK_TOKEN = "<unk>"
    
    BASE_VOCAB = [PAD_TOKEN, CLS_TOKEN, UNK_TOKEN, "A", "C", "G", "T", "N"]

    def __init__(self, max_length: int = 512):
        self.max_length = max_length
        self.vocab: Dict[str, int] = {char: idx for idx, char in enumerate(self.BASE_VOCAB)}
        self.inv_vocab: Dict[int, str] = {idx: char for char, idx in self.vocab.items()}
        self.pad_id = self.vocab[self.PAD_TOKEN]
        self.cls_id = self.vocab[self.CLS_TOKEN]
        self.unk_id = self.vocab[self.UNK_TOKEN]

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def encode(self, sequence: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encodes a DNA sequence string into token IDs and an attention mask.
        
        Args:
            sequence: Raw DNA string (e.g. "GCGGAC...")
            
        Returns:
            token_ids: Tensor of shape [max_length]
            attention_mask: BoolTensor of shape [max_length]
        """
        clean_seq = sequence.upper().strip()
        tokens = [self.cls_id]
        for char in clean_seq:
            tokens.append(self.vocab.get(char, self.unk_id))
            if len(tokens) >= self.max_length:
                break
                
        seq_len = len(tokens)
        mask = [1] * seq_len
        
        if seq_len < self.max_length:
            pad_len = self.max_length - seq_len
            tokens.extend([self.pad_id] * pad_len)
            mask.extend([0] * pad_len)
            
        return torch.tensor(tokens, dtype=torch.long), torch.tensor(mask, dtype=torch.bool)

    def batch_encode(self, sequences: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encodes a batch of DNA sequence strings."""
        batch_ids = []
        batch_masks = []
        for s in sequences:
            ids, mask = self.encode(s)
            batch_ids.append(ids)
            batch_masks.append(mask)
        return torch.stack(batch_ids, dim=0), torch.stack(batch_masks, dim=0)

    def decode(self, token_ids: torch.Tensor) -> str:
        """Decodes token IDs back into a DNA string (skipping special tokens)."""
        chars = []
        for tid in token_ids.tolist():
            if tid in [self.pad_id, self.cls_id]:
                continue
            chars.append(self.inv_vocab.get(tid, "N"))
        return "".join(chars)

    @staticmethod
    def mutate_sequence(sequence: str, position: int, new_nucleotide: str) -> str:
        """Applies a single-nucleotide point mutation at a specific 0-indexed position."""
        if position < 0 or position >= len(sequence):
            raise IndexError(f"Position {position} out of range for sequence of length {len(sequence)}")
        seq_list = list(sequence)
        seq_list[position] = new_nucleotide.upper()
        return "".join(seq_list)

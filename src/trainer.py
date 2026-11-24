"""CPU trainer for Promoter Transformer."""

import time
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
from scipy.stats import pearsonr, spearmanr
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from .promoter_transformer import PromoterTransformerModel
from .config import TRAINING_CONFIG

class PromoterModelTrainer:
    """Trains and evaluates the Promoter Transformer on CPU."""
    
    def __init__(
        self,
        model: PromoterTransformerModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        config: Optional[Dict[str, Any]] = None,
        device: Optional[torch.device] = None
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.config = config or TRAINING_CONFIG
        self.device = device or torch.device("cpu")
        self.model.to(self.device)
        
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.get("learning_rate", 6e-4),
            weight_decay=self.config.get("weight_decay", 1e-4)
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.get("num_epochs", 15),
            eta_min=1e-6
        )
        
        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "val_rmse": [],
            "val_pearson": [],
            "val_spearman": []
        }
        self.best_spearman = -1.0
        self.best_model_weights = None

    def train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        
        for token_ids, mask, targets in self.train_loader:
            token_ids = token_ids.to(self.device)
            mask = mask.to(self.device)
            targets = targets.to(self.device)
            
            self.optimizer.zero_grad()
            preds = self.model(token_ids, mask)
            loss = self.criterion(preds, targets)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
            
        return total_loss / max(n_batches, 1)

    def evaluate(self, loader: DataLoader) -> Dict[str, Any]:
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []
        n_batches = 0
        
        with torch.no_grad():
            for token_ids, mask, targets in loader:
                token_ids = token_ids.to(self.device)
                mask = mask.to(self.device)
                targets = targets.to(self.device)
                
                preds = self.model(token_ids, mask)
                loss = self.criterion(preds, targets)
                
                total_loss += loss.item()
                n_batches += 1
                
                all_preds.extend(preds.squeeze(-1).cpu().numpy().tolist())
                all_targets.extend(targets.squeeze(-1).cpu().numpy().tolist())
                
        preds_arr = np.array(all_preds)
        targets_arr = np.array(all_targets)
        
        mse = float(total_loss / max(n_batches, 1))
        rmse = float(np.sqrt(np.mean((preds_arr - targets_arr) ** 2)))
        
        if len(preds_arr) > 1 and np.std(preds_arr) > 1e-6 and np.std(targets_arr) > 1e-6:
            r_val, _ = pearsonr(preds_arr, targets_arr)
            rho_val, _ = spearmanr(preds_arr, targets_arr)
        else:
            r_val, rho_val = 0.0, 0.0
            
        return {
            "loss": mse,
            "rmse": rmse,
            "pearson": float(r_val),
            "spearman": float(rho_val),
            "preds": preds_arr,
            "targets": targets_arr
        }

    def fit(self, verbose: bool = True) -> Dict[str, Any]:
        num_epochs = self.config.get("num_epochs", 15)
        start_time = time.time()
        
        if verbose:
            print("=" * 72)
            print(f"🚀 Training Promoter Transformer ({num_epochs} Epochs on CPU)")
            print("=" * 72)
            
        for epoch in range(1, num_epochs + 1):
            t0 = time.time()
            train_loss = self.train_epoch()
            val_metrics = self.evaluate(self.val_loader)
            self.scheduler.step()
            elapsed = time.time() - t0
            
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["val_rmse"].append(val_metrics["rmse"])
            self.history["val_pearson"].append(val_metrics["pearson"])
            self.history["val_spearman"].append(val_metrics["spearman"])
            
            if val_metrics["spearman"] > self.best_spearman:
                self.best_spearman = val_metrics["spearman"]
                self.best_model_weights = {
                    k: v.cpu().clone() for k, v in self.model.state_dict().items()
                }
                
            if verbose:
                print(
                    f"Epoch {epoch:02d}/{num_epochs:02d} | "
                    f"Train MSE: {train_loss:.4f} | "
                    f"Val MSE: {val_metrics['loss']:.4f} | "
                    f"RMSE: {val_metrics['rmse']:.3f} | "
                    f"Pearson r: {val_metrics['pearson']:.4f} | "
                    f"Spearman ρ: {val_metrics['spearman']:.4f} | "
                    f"Time: {elapsed:.1f}s"
                )
                
        total_time = time.time() - start_time
        if self.best_model_weights:
            self.model.load_state_dict(self.best_model_weights)
            
        test_metrics = self.evaluate(self.test_loader)
        if verbose:
            print("=" * 72)
            print(f"🏁 Training Complete in {total_time:.1f}s")
            print(f"📊 Final Test Results -> Test MSE: {test_metrics['loss']:.4f} | "
                  f"Spearman ρ: {test_metrics['spearman']:.4f} | Pearson r: {test_metrics['pearson']:.4f}")
            print("=" * 72)
            
        return {
            "history": self.history,
            "test_metrics": test_metrics,
            "total_time_seconds": total_time
        }

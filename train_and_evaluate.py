#!/usr/bin/env python3
"""Automated Training, In Silico Saturation Mutagenesis, and Visualization for Promoter Transformer."""

import os
import argparse
import time
import torch
import numpy as np

from src.config import (
    TERT_PROMOTER_WT,
    TERT_MUTATIONS,
    CANONICAL_PROMOTERS,
    MODEL_CONFIG,
    TRAINING_CONFIG,
    TERT_MUTATION_POS_C228T,
    TERT_MUTATION_POS_C250T
)
from src.dna_tokenizer import DNATokenizer
from src.promoter_transformer import PromoterTransformerModel
from src.data_loader import generate_promoter_benchmark, create_dataloaders
from src.trainer import PromoterModelTrainer
from src.saturation_mutagenesis import InSilicoSaturationMutagenesis
from src.visualizer import PromoterVisualizer

def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate RoPE Promoter Transformer.")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs (default: 15)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training (default: 32)")
    parser.add_argument("--lr", type=float, default=6e-4, help="Learning rate (default: 6e-4)")
    parser.add_argument("--samples", type=int, default=500, help="Number of MPRA promoter samples (default: 500)")
    parser.add_argument("--save-dir", type=str, default="figures", help="Directory for generated figures (default: figures)")
    parser.add_argument("--data-file", type=str, default="data/mpra_promoter_benchmark.csv", help="Path to benchmark CSV")
    parser.add_argument("--ckpt-dir", type=str, default="checkpoints", help="Directory for model weights")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.ckpt_dir, exist_ok=True)
    
    print("=" * 76)
    print("🧬 PROMOTER VARIANT TRANSFORMER: ROTARY POSITION EMBEDDING (RoPE)")
    print("   Avsec et al. (Nature / DeepMind) Enformer Architecture Reproduction")
    print("=" * 76)
    
    # 1. Dataset Generation
    print(f"\n[1/5] Synthesizing MPRA genomic promoter benchmark ({args.samples} samples)...")
    records = generate_promoter_benchmark(
        num_samples=args.samples,
        seq_len=MODEL_CONFIG["max_seq_len"],
        random_seed=42,
        csv_path=args.data_file
    )
    print(f"      Saved curated benchmark to: {args.data_file}")
    
    # 2. Tokenizer & Dataloaders
    print("\n[2/5] Initializing DNA character tokenizer & data loaders...")
    tokenizer = DNATokenizer(max_length=MODEL_CONFIG["max_seq_len"])
    train_loader, val_loader, test_loader = create_dataloaders(
        records=records,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        train_ratio=TRAINING_CONFIG["train_ratio"],
        val_ratio=TRAINING_CONFIG["val_ratio"],
        seed=TRAINING_CONFIG["seed"]
    )
    print(f"      Train samples: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)} | Test: {len(test_loader.dataset)}")
    
    # 3. Model Architecture
    print("\n[3/5] Instantiating RoPE Promoter Transformer Model...")
    cfg = dict(MODEL_CONFIG)
    model = PromoterTransformerModel(cfg)
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"      Total Trainable Parameters: {total_params:,} (~{total_params/1e6:.2f}M)")
    print(f"      Rotary Position Embedding Base: {cfg['rope_theta']}")
    print(f"      Running on Device: CPU (Intel Core i5 Optimized)")
    
    # 4. Training Loop
    print("\n[4/5] Executing high-efficiency training loop...")
    train_cfg = dict(TRAINING_CONFIG)
    train_cfg["num_epochs"] = args.epochs
    train_cfg["learning_rate"] = args.lr
    train_cfg["batch_size"] = args.batch_size
    
    trainer = PromoterModelTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        config=train_cfg,
        device=torch.device("cpu")
    )
    
    train_results = trainer.fit(verbose=True)
    history = train_results["history"]
    test_metrics = train_results["test_metrics"]
    
    best_ckpt_path = os.path.join(args.ckpt_dir, "best_promoter_model.pt")
    torch.save(model.state_dict(), best_ckpt_path)
    print(f"\n      Saved best model checkpoint to: {best_ckpt_path}")
    
    # 5. In Silico Saturation Mutagenesis & Visualizations
    print("\n[5/5] Executing In Silico Saturation Mutagenesis (ISM) & Rendering Figures...")
    visualizer = PromoterVisualizer(output_dir=args.save_dir)
    ism = InSilicoSaturationMutagenesis(model=model, tokenizer=tokenizer)
    
    # Figure A: Predicted vs Actual Scatter Plot
    scatter_path = visualizer.plot_predicted_vs_actual(
        y_true=test_metrics["targets"],
        y_pred=test_metrics["preds"],
        spearman_rho=test_metrics["spearman"],
        pearson_r=test_metrics["pearson"],
        rmse=test_metrics["rmse"],
        filename="promoter_predicted_vs_actual.png"
    )
    print(f"      [Figure 1/3] Sensitivity correlation: {scatter_path}")
    
    # Figure B: In Silico Saturation Mutagenesis Scan over TERT promoter hotspot window
    # Scan from position 210 to 265 (encompassing C228 and C250)
    start_pos = 215
    end_pos = 260
    ism_scan = ism.run_saturation_scan(
        sequence=TERT_PROMOTER_WT,
        start_pos=start_pos,
        end_pos=end_pos,
        batch_size=32
    )
    
    heatmap_path = visualizer.plot_saturation_mutagenesis_heatmap(
        ism_matrix=ism_scan["ism_matrix"],
        positions=ism_scan["positions"],
        ref_bases=ism_scan["ref_bases"],
        filename="tert_saturation_mutagenesis.png"
    )
    print(f"      [Figure 2/3] ISM Hotspot Heatmap:     {heatmap_path}")
    
    # Figure C: TERT Somatic Hotspot Bar Comparison
    # Evaluate WT, C228T, C250T, and benign C228A control
    wt_score = ism.score_single_variant(TERT_PROMOTER_WT, position=TERT_MUTATION_POS_C228T, alt_nucleotide="C")
    c228t_score = ism.score_single_variant(TERT_PROMOTER_WT, position=TERT_MUTATION_POS_C228T, alt_nucleotide="T")
    c250t_score = ism.score_single_variant(TERT_PROMOTER_WT, position=TERT_MUTATION_POS_C250T, alt_nucleotide="T")
    c228a_score = ism.score_single_variant(TERT_PROMOTER_WT, position=TERT_MUTATION_POS_C228T, alt_nucleotide="A")
    
    variants_dict = {
        "TERT Wild-Type": {"activity": wt_score["wt_activity"]},
        "TERT C228T (Glioblastoma)": {"activity": c228t_score["mut_activity"]},
        "TERT C250T (Melanoma)": {"activity": c250t_score["mut_activity"]},
        "C228A (Benign Control)": {"activity": c228a_score["mut_activity"]}
    }
    
    bar_path = visualizer.plot_tert_hotspot_comparison(
        variants_dict=variants_dict,
        filename="tert_hotspot_activity_comparison.png"
    )
    print(f"      [Figure 3/3] Hotspot Bar Comparison: {bar_path}")
    
    print("\n" + "=" * 76)
    print("🏆 ALL EXECUTION MILESTONES COMPLETE")
    print(f"   • Test Spearman Rank Correlation ρ: {test_metrics['spearman']:.4f}")
    print(f"   • Test Pearson Correlation r:       {test_metrics['pearson']:.4f}")
    print(f"   • Test RMSE:                       {test_metrics['rmse']:.3f}")
    print(f"   • TERT C228T Variant Fold Change:   {c228t_score['fold_change']:.2f}x induction")
    print(f"   • Generated Artifacts:              {args.save_dir}/")
    print("=" * 76)

if __name__ == "__main__":
    main()

"""Publication-grade visualization suite for Promoter Variant Transformer."""

import os
from typing import List, Dict, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
from .config import TERT_MUTATION_POS_C228T, TERT_MUTATION_POS_C250T

BG_COLOR = "#0F111A"
CARD_BG = "#1A1D2B"
TEXT_COLOR = "#E6EDF3"
MUTED_TEXT = "#8B949E"
CYAN_ACCENT = "#38BDF8"
GOLD_ACCENT = "#FACC15"
PURPLE_ACCENT = "#C084FC"
GREEN_ACCENT = "#4ADE80"
RED_ACCENT = "#F87171"
GRID_COLOR = "#2D3748"

class PromoterVisualizer:
    """Renders 300 DPI publication-grade figures in 3B1B dark aesthetic."""
    
    def __init__(self, output_dir: str = "figures"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self._apply_style()

    def _apply_style(self):
        plt.rcParams.update({
            "figure.facecolor": BG_COLOR,
            "axes.facecolor": BG_COLOR,
            "axes.edgecolor": "#30363D",
            "axes.labelcolor": TEXT_COLOR,
            "xtick.color": TEXT_COLOR,
            "ytick.color": TEXT_COLOR,
            "text.color": TEXT_COLOR,
            "grid.color": GRID_COLOR,
            "grid.linestyle": "--",
            "grid.alpha": 0.4,
            "font.family": "serif",
            "font.size": 11,
        })

    def plot_saturation_mutagenesis_heatmap(
        self,
        ism_matrix: np.ndarray,
        positions: List[int],
        ref_bases: List[str],
        filename: str = "tert_saturation_mutagenesis.png"
    ) -> str:
        """Plots the 4 x L In Silico Saturation Mutagenesis heatmap highlighting C228T and C250T."""
        fig, ax = plt.subplots(figsize=(14, 5), dpi=300)
        
        # Diverging colormap centered at 0
        vmax = max(abs(ism_matrix.min()), abs(ism_matrix.max()))
        vmax = max(vmax, 0.5)
        
        cax = ax.imshow(
            ism_matrix,
            cmap="coolwarm",
            aspect="auto",
            vmin=-vmax,
            vmax=vmax,
            interpolation="nearest"
        )
        
        # Colorbar
        cbar = fig.colorbar(cax, ax=ax, fraction=0.02, pad=0.03)
        cbar.ax.yaxis.set_tick_params(color=TEXT_COLOR)
        cbar.set_label("Variant Effect $\\Delta$Activity (Mutant - WT)", color=TEXT_COLOR, size=11, labelpad=10)
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color=TEXT_COLOR)
        
        # Nucleotides on y-axis
        ax.set_yticks(np.arange(4))
        ax.set_yticklabels(["A", "C", "G", "T"], fontsize=11, fontweight="bold", color=GOLD_ACCENT)
        
        # Positions on x-axis (sampled every 5 positions for readability)
        step = max(1, len(positions) // 15)
        tick_indices = list(range(0, len(positions), step))
        ax.set_xticks(tick_indices)
        ax.set_xticklabels([f"{positions[i]} ({ref_bases[i]})" for i in tick_indices], fontsize=8, rotation=45, color=TEXT_COLOR)
        
        # Highlight C228 and C250 positions if present in the window
        for idx, pos in enumerate(positions):
            if pos == TERT_MUTATION_POS_C228T:
                ax.axvline(idx, color=GOLD_ACCENT, linestyle="--", linewidth=1.8, alpha=0.9)
                ax.text(idx, -0.6, "C228T (Hotspot)", color=GOLD_ACCENT, fontsize=9, fontweight="bold", ha="center")
            elif pos == TERT_MUTATION_POS_C250T:
                ax.axvline(idx, color=CYAN_ACCENT, linestyle="--", linewidth=1.8, alpha=0.9)
                ax.text(idx, -0.6, "C250T (Hotspot)", color=CYAN_ACCENT, fontsize=9, fontweight="bold", ha="center")
                
        ax.set_title("In Silico Saturation Mutagenesis: Human TERT Core Promoter", fontsize=13, fontweight="bold", color=GOLD_ACCENT, pad=22)
        ax.set_xlabel("Genomic Position (GRCh38 Chr5 Window) & Reference Nucleotide", fontsize=11, color=TEXT_COLOR, labelpad=8)
        ax.set_ylabel("Mutant Base", fontsize=11, color=TEXT_COLOR, labelpad=8)
        
        plt.tight_layout()
        out_path = os.path.join(self.output_dir, filename)
        plt.savefig(out_path, dpi=300, facecolor=BG_COLOR)
        plt.close(fig)
        return out_path

    def plot_tert_hotspot_comparison(
        self,
        variants_dict: Dict[str, Dict[str, float]],
        filename: str = "tert_hotspot_activity_comparison.png"
    ) -> str:
        """Bar plot comparing TERT Wild-Type vs C228T, C250T, and benign control."""
        fig, ax = plt.subplots(figsize=(9, 6), dpi=300)
        
        names = list(variants_dict.keys())
        activities = [variants_dict[k]["activity"] for k in names]
        colors = [
            "#64748B",       # WT (slate grey)
            GOLD_ACCENT,     # C228T (gold)
            CYAN_ACCENT,     # C250T (cyan)
            PURPLE_ACCENT    # Control (purple)
        ][:len(names)]
        
        bars = ax.bar(names, activities, color=colors, width=0.55, edgecolor="#FFFFFF", linewidth=0.5, zorder=3)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        
        # Value callouts on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.2f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center", va="bottom",
                fontsize=10, fontweight="bold",
                color=TEXT_COLOR
            )
            
        ax.set_title("Human TERT Promoter: Wild-Type vs Oncogenic Somatic Driver Mutations", fontsize=12, fontweight="bold", color=GOLD_ACCENT, pad=15)
        ax.set_ylabel("Predicted Promoter Transcription Activity", fontsize=11, color=TEXT_COLOR, labelpad=10)
        ax.set_ylim(0, max(activities) * 1.25)
        
        # Subtitle callout
        ax.text(
            0.5, 0.88,
            "C228T & C250T generate de novo GABP/ETS binding motifs (CCGGAA),\ninducing massive telomerase reactivation in 70% of glioblastomas & melanomas.",
            transform=ax.transAxes,
            ha="center",
            fontsize=9,
            style="italic",
            color=MUTED_TEXT,
            bbox=dict(boxstyle="round,pad=0.5", facecolor=CARD_BG, edgecolor="#30363D", alpha=0.8)
        )
        
        plt.tight_layout()
        out_path = os.path.join(self.output_dir, filename)
        plt.savefig(out_path, dpi=300, facecolor=BG_COLOR)
        plt.close(fig)
        return out_path

    def plot_predicted_vs_actual(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        spearman_rho: float,
        pearson_r: float,
        rmse: float,
        filename: str = "promoter_predicted_vs_actual.png"
    ) -> str:
        """Plots experimental vs model-predicted transcription levels."""
        fig, ax = plt.subplots(figsize=(8, 7), dpi=300)
        
        errors = np.abs(y_true - y_pred)
        ax.scatter(
            y_true,
            y_pred,
            c=errors,
            cmap="plasma",
            alpha=0.85,
            edgecolors="#FFFFFF",
            linewidths=0.4,
            s=55,
            zorder=3
        )
        
        min_val = min(y_true.min(), y_pred.min()) - 0.2
        max_val = max(y_true.max(), y_pred.max()) + 0.2
        ax.plot([min_val, max_val], [min_val, max_val], color=GOLD_ACCENT, linestyle="--", linewidth=1.5, label="Identity Line ($y = x$)")
        
        poly_fit = np.polyfit(y_true, y_pred, 1)
        fit_fn = np.poly1d(poly_fit)
        x_vals = np.linspace(min_val, max_val, 100)
        ax.plot(x_vals, fit_fn(x_vals), color=CYAN_ACCENT, linewidth=1.8, label=f"Fit: $y = {poly_fit[0]:.2f}x + {poly_fit[1]:.2f}$")
        
        ax.set_xlim(min_val, max_val)
        ax.set_ylim(min_val, max_val)
        ax.grid(True, linestyle="--", alpha=0.3)
        
        stats_text = (
            f"Spearman $\\rho$: {spearman_rho:.4f}\n"
            f"Pearson $r$: {pearson_r:.4f}\n"
            f"RMSE: {rmse:.3f}"
        )
        ax.text(
            0.05, 0.93,
            stats_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.6", facecolor=CARD_BG, edgecolor=CYAN_ACCENT, alpha=0.9),
            color=TEXT_COLOR
        )
        
        ax.set_title("RoPE Promoter Transformer: MPRA Sensitivity Benchmark", fontsize=13, fontweight="bold", color=GOLD_ACCENT, pad=15)
        ax.set_xlabel("Ground Truth Promoter Transcription Activity", fontsize=11, color=TEXT_COLOR, labelpad=8)
        ax.set_ylabel("Transformer Predicted Activity", fontsize=11, color=TEXT_COLOR, labelpad=8)
        ax.legend(loc="lower right", facecolor=CARD_BG, edgecolor="#30363D", fontsize=9)
        
        plt.tight_layout()
        out_path = os.path.join(self.output_dir, filename)
        plt.savefig(out_path, dpi=300, facecolor=BG_COLOR)
        plt.close(fig)
        return out_path

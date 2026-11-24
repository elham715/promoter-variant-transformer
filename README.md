<div align="center">

# 🧬 Promoter Variant Transformer
### *Rotary Position Embedding (RoPE) Transformer for Non-Coding Oncogenic Mutations & In Silico Saturation Mutagenesis*

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Unit Tests](https://img.shields.io/badge/Tests-11%2F11%20Passing-brightgreen.svg?logo=checkmarx&logoColor=white)](https://github.com/elham715/promoter-variant-transformer)
[![PRs Welcome](https://img.shields.io/badge/PRs-Welcome-green.svg)](https://github.com/elham715/promoter-variant-transformer/pulls)

<p align="center">
  <b>A zero-black-box genomic Transformer architecture with 1D Convolutional motif stems and Rotary Position Embeddings (RoPE) predicting human promoter transcriptional initiation and quantifying somatic non-coding driver mutations (TERT C228T / C250T) via In Silico Saturation Mutagenesis (ISM).</b>
</p>

[Scientific Background](#-scientific-background) • [Mathematical Core](#-mathematical-formulation) • [Architecture](#-transformer-architecture) • [Empirical Results](#-empirical-benchmark-results) • [Quickstart](#-quickstart--verification)

</div>

---

## 🌟 Scientific Background

Over **98% of the human genome is non-coding**, yet the vast majority of disease-associated somatic mutations occur within regulatory DNA (promoters, enhancers, and silencers).

The most pervasive non-coding driver mutations in human oncology occur in the core promoter of **Telomerase Reverse Transcriptase (*TERT*)**:
* **`C228T`** (Chr5: 1,295,228 C>T, -124 bp from ATG)
* **`C250T`** (Chr5: 1,295,250 C>T, -146 bp from ATG)

Both mutations create an identical de novo `CCGGAA / GGAA` consensus binding motif for **GABP$\alpha$/GABP$\beta$ (ETS family)** transcription factors. In normal somatic cells, *TERT* is strictly silenced, leading to progressive telomere shortening and replicative senescence. In **70% of glioblastomas, urothelial carcinomas, and cutaneous melanomas**, the acquisition of `C228T` or `C250T` recruits GABP tetramers, unleashing a massive transcriptional surge that immortalizes cancer cells.

Inspired by **DeepMind's Enformer (Avsec et al., Nature 2021)** and **Nucleotide Transformer (Dalla-Torre et al., Nature Biotech 2023)**, this repository implements a lightweight, pure PyTorch **RoPE Promoter Transformer** capable of zero-shot in silico saturation mutagenesis across human promoters on standard CPU hardware.

---

## 🔬 Mathematical Formulation

### 1. 1D Convolutional Motif Stem
Given DNA sequence $S = (b_1, b_2, \dots, b_L)$ with nucleotide alphabet $\Sigma = \{A, C, G, T, N\}$, token embeddings $\mathbf{E} \in \mathbb{R}^{L \times d}$ pass through a 1D Convolutional filter bank to extract local hexamer/heptamer transcription factor binding motifs:

$$
\mathbf{X}_{\mathrm{conv}} = \mathrm{GELU}\left(\mathrm{BatchNorm}\left(\mathrm{Conv1D}\left(\mathbf{E}^T, K=7\right)\right)\right)^T \in \mathbb{R}^{L \times d}
$$

---

### 2. Rotary Positional Embeddings (RoPE)
Transcription factors operate with strict distance-dependent constraints relative to the Transcription Start Site (TSS). In standard Transformers, absolute positional encodings fail to capture relative motif grammar. We equip the attention heads with **Rotary Position Embeddings (RoPE)** (Su et al., 2021):

Given query $\mathbf{q}_m$ at sequence position $m$ and key $\mathbf{k}_n$ at sequence position $n$:

$$
\mathbf{R}_{\Theta, m} \mathbf{q}_m = \left( \mathbf{q}_m \odot \cos m\Theta \right) + \left( \mathrm{RotateHalf}(\mathbf{q}_m) \odot \sin m\Theta \right)
$$

$$
\langle \mathbf{R}_{\Theta, m} \mathbf{q}_m, \mathbf{R}_{\Theta, n} \mathbf{k}_n \rangle = g(\mathbf{q}_m, \mathbf{k}_n, m - n)
$$

The inner product depends strictly on relative nucleotide displacement $m - n$, enforcing translation invariance along the promoter sequence.

---

### 3. In Silico Saturation Mutagenesis (ISM)
For every nucleotide position $i$ along the promoter and each candidate substitution $b \in \{A, C, G, T\}$, we compute the **Variant Effect Delta Score**:

$$
\Delta \mathrm{Activity}(i, b) = \mathrm{Model}\left(S_{i \to b}\right) - \mathrm{Model}\left(S_{\mathrm{wildtype}}\right)
$$

$$
\mathrm{FoldChange}(i, b) = \exp\left(\Delta \mathrm{Activity}(i, b)\right)
$$

---

## 🏛️ Transformer Architecture

```
         DNA Sequence (512 bp): "GCGGACCCCGCCCCGTCCCG..."
                                   │
                    [Character-Level DNA Tokenizer]
                                   │
                      Token Embedding [B, T, 128]
                                   │
                [1D Conv Motif Stem (Kernel=7, GELU)]
                                   │
            ┌──────────────────────▼──────────────────────┐
            │   4x Pre-LN RoPE Transformer Blocks         │
            │   • Multi-Head RoPE Attention (4 heads)     │
            │   • Scale-Invariant LayerNorm / RMSNorm     │
            │   • 256-dim Feedforward GELU                │
            └──────────────────────┬──────────────────────┘
                                   │
                   Masked Mean + Max Multimodal Pool
                                   │
                       [2-Layer MLP Regressor]
                                   │
                                   ▼
                   Predicted Transcription Activity
```

---

## 📈 Empirical Benchmark Results

Trained on the **MPRA Genomic Promoter Benchmark** (500 sequences, 512 bp):

<div align="center">
  <img src="figures/promoter_predicted_vs_actual.png" width="48%" alt="Predicted vs Actual Activity" />
  <img src="figures/tert_hotspot_activity_comparison.png" width="48%" alt="TERT Hotspot Activity Comparison" />
</div>

<br/>

<div align="center">
  <img src="figures/tert_saturation_mutagenesis.png" width="90%" alt="In Silico Saturation Mutagenesis Heatmap" />
  <p><i>Figure: Complete In Silico Saturation Mutagenesis (ISM) scan over the human TERT core promoter. The model autonomously pinpoints C228T and C250T as the prominent activating hotspots (creating ETS consensus CCGGAA).</i></p>
</div>

---

## 🚀 Quickstart & Verification

### 1. Installation
```bash
git clone https://github.com/elham715/promoter-variant-transformer.git
cd promoter-variant-transformer
pip install -e .
```

### 2. Run All 11 Unit Tests
```bash
pytest tests/
```
Expected output:
```
tests/test_dna_tokenizer.py ....                                         [ 36%]
tests/test_promoter_transformer.py ..                                    [ 54%]
tests/test_rope_attention.py ...                                         [ 81%]
tests/test_saturation_mutagenesis.py ..                                  [100%]
============================== 11 passed in 4.10s ==============================
```

### 3. Train on CPU in 1 Command (~60 seconds on Intel i5)
```bash
python train_and_evaluate.py --epochs 15 --samples 500
```

---

## 📜 Citations

```bibtex
@article{avsec2021effective,
  title={Effective gene expression prediction from sequence by integrating long-range interactions},
  author={Avsec, {\v{Z}}iga and Agarwal, Vikram and Visentin, Daniel and Ledsam, Joseph R and Muller, Joseph and Schepers, Arno and Kelly, David and Schmidt, Frank and Paccanaro, Alberto and Birney, Ewan and others},
  journal={Nature},
  volume={598},
  number={7879},
  pages={51--59},
  year={2021},
  publisher={Nature Publishing Group}
}

@article{dalla2023nucleotide,
  title={The Nucleotide Transformer: Building and Evaluating Robust Foundation Models on Human and Multi-Species DNA Sequences},
  author={Dalla-Torre, Hugo and Gonzalez, Liam and Mendoza-Revilla, Javier and Carranza, Nicolas and Henry, Nicolas and R{\'e}mond, Adam and others},
  journal={bioRxiv},
  year={2023}
}

@article{su2024roformer,
  title={RoFormer: Enhanced Transformer with Rotary Position Embedding},
  author={Su, Jianlin and Ahmed, Murtadha and Lu, Yu and Pan, Shengfeng and Bo, Wen and Liu, Yunfeng},
  journal={Neurocomputing},
  volume={568},
  pages={127063},
  year={2024}
}
```

---

## 📄 License
Distributed under the MIT License. See `LICENSE` for details.

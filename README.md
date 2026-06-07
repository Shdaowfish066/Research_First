# Sentiment Classification with LSTM — Replication & Extension

Replication and extension of the IJARCCE 2025 paper:

> **"Comparative Analysis of Activation Functions in LSTM Models for Sentiment Classification on IMDb Dataset"**
> IJARCCE Vol. 14, Issue 4, April 2025 · DOI: [10.17148/IJARCCE.2025.14421](https://doi.org/10.17148/IJARCCE.2025.14421)

All experiments use the [IMDb Large Movie Review Dataset](https://ai.stanford.edu/~amaas/data/sentiment/) and are implemented in **pure PyTorch**.

---

## Repository Structure

```
Research/
├── dataset/
│   └── IMDB Dataset.csv          # Raw IMDb 50k review dataset
│
├── Paper/
│   └── IJARCCE.2025.14421.pdf    # Reference paper
│
├── something.py                  # Replication of the paper's three LSTM models
├── experiment.py                 # Experimental models (ExpA / ExpB / ExpC)
├── activation_sweep.py           # Full 10×10 activation-function sweep
│
└── results/
    ├── result_original/          # Paper replication results
    │   ├── LSTM1_LeakyReLU_Tanh.pt
    │   ├── LSTM2_ELU_Tanh.pt
    │   ├── LSTM3_ReLU_ELU.pt
    │   ├── results_summary.csv
    │   └── *.png                 # Training curves, metric comparison, confusion matrices, radar chart, table
    │
    ├── result_activation_sweep/  # Activation sweep results
    │   ├── result_activation_sweep.csv
    │   └── *.png                 # Heatmaps, bar charts, per-activation averages
    │
    └── result_experimental/      # Experimental model results
        ├── ExpA_BiLSTM_SelfAttention.pt
        ├── ExpB_CNN_BiLSTM_Attention.pt
        ├── ExpC_StackedBiLSTM_MultiHeadAttn.pt
        ├── result_experimental.csv
        └── *.png                 # Training curves, paper vs. experimental comparison, confusion matrices, radar chart
```

---

## Experiments

### 1 · Paper Replication (`something.py`)

Re-implements the three Bi-LSTM models from the paper using identical architecture and hyper-parameters.

| Model | Activation 1 | Activation 2 | Paper Acc  | Our Acc |
| ----- | ------------ | ------------ | ---------- | ------- |
| LSTM1 | LeakyReLU    | Tanh         | 88.3 %     | 87.54 % |
| LSTM2 | ELU          | Tanh         | 88.3 %     | 86.00 % |
| LSTM3 | ReLU         | ELU          | **88.4 %** | 86.88 % |

Architecture fixed to paper spec: `Vocab=5 000 · MaxLen=200 · EmbedDim=128 · BiLSTM(64) → LSTM(32) → Dense(64) → Dense(32) → Output`

---

### 2 · Activation Function Sweep (`activation_sweep.py`)

Exhaustive **10 × 10 = 100 combination** grid search over activation functions in the two dense layers, keeping all other hyper-parameters identical to the paper.

**Activation functions tested:** ReLU, ELU, LeakyReLU, Tanh, GELU, SiLU (Swish), Mish, SELU, PReLU, Hardswish

Top result from sweep: **Mish + PReLU** — Acc 87.66 %, F1 87.51 %

Results saved to `results/result_activation_sweep/result_activation_sweep.csv`.

---

### 3 · Experimental Models (`experiment.py`)

Three architecturally enhanced models designed to surpass the paper's best accuracy of 88.4 %.

| Model              | Architecture                          | Acc         | Precision | Recall  | F1          |
| ------------------ | ------------------------------------- | ----------- | --------- | ------- | ----------- |
| Paper best (LSTM3) | BiLSTM + Dense                        | 88.4 %      | 90.3 %    | 86.3 %  | 88.3 %      |
| **ExpA**           | BiLSTM + Self-Attention               | **90.20 %** | 90.79 %   | 89.48 % | 90.13 %     |
| ExpB               | CNN + BiLSTM + Attention              | 88.97 %     | 89.63 %   | 88.14 % | 88.88 %     |
| **ExpC**           | Stacked BiLSTM + Multi-Head Attention | **90.15 %** | 89.84 %   | 90.54 % | **90.19 %** |

Key improvements over the paper:

- Vocabulary: 5 k → 15 k
- Sequence length: 200 → 300
- Embedding dimension: 128 → 256
- BiLSTM units: 64 → 128
- Self-Attention / Multi-Head Attention layers
- Conv1D feature extraction (ExpB)
- Label smoothing loss
- Gradient clipping
- Cosine annealing LR schedule

---

## Setup

```bash
# Create virtual environment
python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate  # Linux / macOS

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install numpy pandas matplotlib seaborn scikit-learn
```

> **Hardware:** Experiments were run on an NVIDIA RTX 5080 (CUDA 12). CPU execution is supported but significantly slower.

---

## Running the Scripts

```bash
# 1 — Replicate the paper's three models
python something.py

# 2 — Run the full activation-function sweep (100 combinations)
python activation_sweep.py

# 3 — Train experimental models (ExpA / ExpB / ExpC)
python experiment.py
```

All outputs (`.pt` checkpoints, `.csv` metrics, `.png` plots) are written to the respective subdirectory under `results/`.

---

## Results Summary

| Experiment              | Best Accuracy | vs. Paper (+Δ) |
| ----------------------- | ------------- | -------------- |
| Paper replication       | 87.54 %       | − 0.86 pp      |
| Activation sweep        | 87.66 %       | − 0.74 pp      |
| **Experimental (ExpA)** | **90.20 %**   | **+1.80 pp**   |
| **Experimental (ExpC)** | **90.15 %**   | **+1.75 pp**   |

Both ExpA and ExpC **exceed the paper's best reported accuracy of 88.4 %**.

---

## Citation

```bibtex
@article{ijarcce2025lstm,
  title   = {Comparative Analysis of Activation Functions in LSTM Models
             for Sentiment Classification on IMDb Dataset},
  journal = {IJARCCE},
  volume  = {14},
  number  = {4},
  year    = {2025},
  doi     = {10.17148/IJARCCE.2025.14421}
}
```

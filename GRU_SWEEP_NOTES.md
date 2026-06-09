# GRU Activation Sweep — Context & Reference Notes

## Source Paper

**"Comparative Analysis of Activation Functions in LSTM Models for Sentiment Classification on IMDb Dataset"**
IJARCCE Vol. 14, Issue 4, April 2025 · DOI: 10.17148/IJARCCE.2025.14421

---

## Paper Architecture (Table 1 reproduction)

```
Input(200) → Embedding(5000, 128) → BiLSTM(64, return_seq=True)
           → LSTM(32) → Dense(64)+Act1 → Dense(32)+Act2 → Dense(1, sigmoid)
```

**Paper results (Table 1):**

| Model     | Act1      | Act2    | Acc (%)  | Prec (%) | Recall (%) | F1 (%)   |
| --------- | --------- | ------- | -------- | -------- | ---------- | -------- |
| LSTM1     | LeakyReLU | Tanh    | 88.3     | 86.1     | 91.7       | 88.8     |
| LSTM2     | ELU       | Tanh    | 88.3     | 88.7     | 88.0       | 88.3     |
| **LSTM3** | **ReLU**  | **ELU** | **88.4** | **90.3** | **86.3**   | **88.3** |

---

## Hyperparameters (paper-exact, used in both LSTM and GRU sweeps)

| Parameter   | Value                                                   |
| ----------- | ------------------------------------------------------- |
| Vocab size  | 5,000                                                   |
| Max length  | 200                                                     |
| Embed dim   | 128                                                     |
| BiRNN units | 64                                                      |
| RNN units 2 | 32                                                      |
| Dense 1     | 64                                                      |
| Dense 2     | 32                                                      |
| Dropout     | 0.4                                                     |
| Batch size  | 64                                                      |
| Epochs      | 10 (early stop patience=3)                              |
| Optimizer   | Adam (lr=1e-3)                                          |
| Scheduler   | ReduceLROnPlateau (factor=0.5, patience=2, min_lr=1e-6) |
| Loss        | BCEWithLogitsLoss                                       |
| Grad clip   | 1.0                                                     |

---

## Activation Functions Tested (10 each → 100 combos)

| Name      | Type           | Notes                      |
| --------- | -------------- | -------------------------- |
| ReLU      | Paper original | max(0, x)                  |
| ELU       | Paper original | α=1.0                      |
| LeakyReLU | Paper original | α=0.01                     |
| Tanh      | Paper original | classic saturation         |
| GELU      | Modern         | Gaussian error linear unit |
| SiLU      | Modern         | Swish; x·σ(x)              |
| Mish      | Modern         | x·tanh(softplus(x))        |
| SELU      | Modern         | self-normalizing           |
| PReLU     | Modern         | learnable negative slope   |
| Hardswish | Modern         | piecewise approximation    |

---

## LSTM Sweep Results (activation_sweep.py)

**File:** `results/result_activation_sweep/result_activation_sweep.csv`

| Rank | Act1      | Act2      | Acc (%) | F1 (%) | Prec (%) | Recall (%) | Beat Paper? |
| ---- | --------- | --------- | ------- | ------ | -------- | ---------- | ----------- |
| 1    | Mish      | PReLU     | 87.66   | 87.51  | 88.62    | 86.42      | No          |
| 2    | Hardswish | Mish      | 87.61   | 87.52  | 88.18    | 86.86      | No          |
| 3    | ReLU      | LeakyReLU | 87.56   | 87.53  | 87.74    | 87.32      | No          |

**Summary:** 0/100 LSTM combos beat the paper's 88.4%. Best was Mish+PReLU at 87.66%.

---

## GRU Sweep (gru_activation_sweep.py)

**File:** `results/GRU_Results/result_gru_activation_sweep.csv`

### Architecture Change (only difference from LSTM sweep)

| Component         | LSTM version                           | GRU version                           |
| ----------------- | -------------------------------------- | ------------------------------------- |
| First RNN layer   | `nn.LSTM(128, 64, bidirectional=True)` | `nn.GRU(128, 64, bidirectional=True)` |
| Second RNN layer  | `nn.LSTM(128, 32)`                     | `nn.GRU(128, 32)`                     |
| Hidden extraction | `_, (h, _) = self.lstm(o)` → `h[-1]`   | `_, h = self.gru(o)` → `h[-1]`        |

**Key GRU property:** GRU has no separate cell state — `forward()` returns `(output, hidden)`.
For single-layer unidirectional GRU, `hidden` shape = `(1, batch, hidden_size)`.

### Output Files

| File                              | Description                                                        |
| --------------------------------- | ------------------------------------------------------------------ |
| `result_gru_activation_sweep.csv` | 100-row sweep results (Act1, Act2, Acc, Prec, Rec, F1, Beat_Paper) |
| `01_accuracy_heatmap.png`         | 10×10 accuracy grid; blue border = beats 88.4%                     |
| `02_f1_heatmap.png`               | 10×10 F1 score grid                                                |
| `03_top15_bar.png`                | Bar chart of top 15 combinations                                   |
| `04_per_activation_avg.png`       | Average accuracy per activation in Act1/Act2 position              |
| `05_precision_heatmap.png`        | 10×10 precision grid                                               |
| `06_recall_heatmap.png`           | 10×10 recall grid                                                  |
| `07_confusion_matrix_top5.png`    | Confusion matrices for top-5 combos (re-trained)                   |
| `08_gru_vs_lstm_comparison.png`   | Side-by-side LSTM vs GRU top-10 bar charts                         |

### GRU Results (top 10, sorted by accuracy)

| Rank | Act1      | Act2  | Acc (%) | Prec (%) | Recall (%) | F1 (%) | Beat Paper? |
| ---- | --------- | ----- | ------- | -------- | ---------- | ------ | ----------- |
| 1    | LeakyReLU | Tanh  | 88.05   | 88.4     | 87.6       | 88.00  | No          |
| 2    | PReLU     | ELU   | 88.04   | 87.7     | 88.4       | 88.09  | No          |
| 3    | GELU      | SiLU  | 88.03   | 88.2     | 87.8       | 88.00  | No          |
| 4    | ELU       | PReLU | 88.02   | 87.8     | 88.3       | 88.05  | No          |
| 5    | ELU       | GELU  | 88.01   | 88.3     | 87.6       | 87.96  | No          |
| 6    | LeakyReLU | SELU  | 88.00   | 88.0     | 88.0       | 88.00  | No          |
| 7    | ELU       | ReLU  | 87.99   | 87.0     | 89.4       | 88.16  | No          |
| 8    | Tanh      | GELU  | 87.98   | 86.8     | 89.6       | 88.17  | No          |
| 9    | ELU       | SELU  | 87.96   | 88.5     | 87.3       | 87.88  | No          |
| 10   | Mish      | GELU  | 87.96   | 87.0     | 89.3       | 88.12  | No          |

**Summary:** 0/100 GRU combos beat the paper's 88.4%. Best was LeakyReLU+Tanh at 88.05% (0.35% below paper).
**Sweep runtime:** 21.4 minutes on RTX 5080.

---

---

## FullBiLSTM Sweep (bilstm_bigru_sweep.py)

Both LSTM layers bidirectional. Dense1 input = 32×2 = 64 (concat fwd+bwd hidden from 2nd BiLSTM).

**File:** `results/BiLSTM_Results/result_fullbilstm_sweep.csv`

### Architecture

```
Embedding → BiLSTM(64) → BiLSTM(32) → cat(h_fwd, h_bwd)=64 → Dense(64)+Act1 → Dense(32)+Act2 → Out
```

### Top 10 FullBiLSTM Results

| Rank | Act1      | Act2      | Acc (%) | Prec (%) | Recall (%) | F1 (%) | Beat Paper? |
| ---- | --------- | --------- | ------- | -------- | ---------- | ------ | ----------- |
| 1    | Tanh      | GELU      | 87.82   | 87.6     | 88.1       | 87.85  | No          |
| 2    | PReLU     | Mish      | 87.53   | 87.9     | 87.0       | 87.47  | No          |
| 3    | LeakyReLU | PReLU     | 87.51   | 87.8     | 87.1       | 87.45  | No          |
| 4    | SELU      | Tanh      | 87.50   | 87.2     | 88.0       | 87.56  | No          |
| 5    | ReLU      | ELU       | 87.46   | 85.5     | 90.2       | 87.80  | No          |
| 6    | Hardswish | ELU       | 87.45   | 85.3     | 90.4       | 87.81  | No          |
| 7    | ELU       | ReLU      | 87.42   | 86.6     | 88.5       | 87.56  | No          |
| 8    | SELU      | Hardswish | 87.41   | 86.9     | 88.2       | 87.50  | No          |
| 9    | GELU      | Tanh      | 87.39   | 86.7     | 88.3       | 87.50  | No          |
| 10   | LeakyReLU | ReLU      | 87.38   | 88.7     | 85.7       | 87.16  | No          |

**Summary:** 0/100 FullBiLSTM combos beat the paper. Best was Tanh+GELU at 87.82% (−0.58% vs paper). **Slower and worse** than single-directional second layer — making the second layer bidirectional hurts for this task.
**Sweep runtime:** 31.5 minutes on RTX 5080.

### FullBiLSTM Output Files

| File                                | Description                          |
| ----------------------------------- | ------------------------------------ |
| `result_fullbilstm_sweep.csv`       | 100-row sweep results                |
| `01_accuracy_heatmap.png`           | 10×10 accuracy grid                  |
| `02_f1_heatmap.png`                 | 10×10 F1 grid                        |
| `03_top15_bar.png`                  | Top-15 bar chart                     |
| `04_per_activation_avg.png`         | Per-activation averages              |
| `05_precision_heatmap.png`          | 10×10 precision grid                 |
| `06_recall_heatmap.png`             | 10×10 recall grid                    |
| `07_confusion_matrix_top5.png`      | Confusion matrices top-5             |
| `08_comparison_chart.png`           | vs LSTM + GRU sweeps                 |
| `09_bilstm_vs_bigru_comparison.png` | FullBiLSTM vs FullBiGRU side-by-side |

---

## FullBiGRU Sweep (bilstm_bigru_sweep.py)

Both GRU layers bidirectional. Same structure as FullBiLSTM but with GRU cells.

**File:** `results/BiGRU_Results/result_fullbigru_sweep.csv`

### Architecture

```
Embedding → BiGRU(64) → BiGRU(32) → cat(h_fwd, h_bwd)=64 → Dense(64)+Act1 → Dense(32)+Act2 → Out
```

### Top 10 FullBiGRU Results

| Rank | Act1      | Act2      | Acc (%) | Prec (%) | Recall (%) | F1 (%)    | Beat Paper? |
| ---- | --------- | --------- | ------- | -------- | ---------- | --------- | ----------- |
| 1    | Tanh      | Hardswish | 88.06   | 87.5     | 88.9       | 88.15     | No          |
| 2    | ReLU      | Mish      | 88.03   | 86.9     | 89.5       | 88.21     | No          |
| 3    | Hardswish | Mish      | 88.02   | 87.0     | 89.4       | 88.19     | No          |
| 4    | SiLU      | SiLU      | 87.89   | 85.5     | 91.3       | **88.29** | No          |
| 5    | GELU      | SELU      | 87.88   | 87.9     | 87.8       | 87.88     | No          |
| 6    | LeakyReLU | Hardswish | 87.88   | 86.2     | 90.3       | 88.16     | No          |
| 7    | GELU      | SiLU      | 87.86   | 87.5     | 88.3       | 87.92     | No          |
| 8    | GELU      | ELU       | 87.85   | 87.4     | 88.5       | 87.93     | No          |
| 9    | ReLU      | SiLU      | 87.84   | 88.4     | 87.1       | 87.75     | No          |
| 10   | SELU      | SELU      | 87.82   | 86.1     | 90.3       | 88.11     | No          |

**Summary:** 0/100 FullBiGRU combos beat the paper. Best was Tanh+Hardswish at 88.06% (−0.34% vs paper). Best F1 overall: SiLU+SiLU at 88.29%.
**Sweep runtime:** 21.8 minutes on RTX 5080.

### FullBiGRU Output Files

| File                           | Description              |
| ------------------------------ | ------------------------ |
| `result_fullbigru_sweep.csv`   | 100-row sweep results    |
| `01_accuracy_heatmap.png`      | 10×10 accuracy grid      |
| `02_f1_heatmap.png`            | 10×10 F1 grid            |
| `03_top15_bar.png`             | Top-15 bar chart         |
| `04_per_activation_avg.png`    | Per-activation averages  |
| `05_precision_heatmap.png`     | 10×10 precision grid     |
| `06_recall_heatmap.png`        | 10×10 recall grid        |
| `07_confusion_matrix_top5.png` | Confusion matrices top-5 |
| `08_comparison_chart.png`      | vs LSTM + GRU sweeps     |

---

## Master Comparison Table (all experiments)

| Experiment  | Script                  | Best Combo     | Acc (%)  | Prec (%) | Rec (%) | F1 (%) | vs Paper |
| ----------- | ----------------------- | -------------- | -------- | -------- | ------- | ------ | -------- |
| Paper LSTM1 | —                       | LeakyReLU+Tanh | 88.3     | 86.1     | 91.7    | 88.8   | baseline |
| Paper LSTM3 | —                       | ReLU+ELU       | **88.4** | 90.3     | 86.3    | 88.3   | baseline |
| LSTM Sweep  | activation_sweep.py     | Mish+PReLU     | 87.66    | 88.6     | 86.4    | 87.51  | −0.74%   |
| GRU Sweep   | gru_activation_sweep.py | LeakyReLU+Tanh | 88.05    | 88.4     | 87.6    | 88.00  | −0.35%   |
| FullBiLSTM  | bilstm_bigru_sweep.py   | Tanh+GELU      | 87.82    | 87.6     | 88.1    | 87.85  | −0.58%   |
| FullBiGRU   | bilstm_bigru_sweep.py   | Tanh+Hardswish | 88.06    | 87.5     | 88.9    | 88.15  | −0.34%   |

**Key finding:** GRU consistently outperforms LSTM in pure activation sweeps. FullBiGRU is the best purely architectural variant at 88.06%.

---

## How to Run

```powershell
cd D:\Research
& .\.venv\Scripts\Activate.ps1

# GRU sweep (100 combos, ~21 min)
python gru_activation_sweep.py

# FullBiLSTM + FullBiGRU sweeps (200 combos, ~53 min)
python bilstm_bigru_sweep.py
```

---

## Metrics Definition (from paper)

- **Accuracy** = (TP + TN) / (TP + TN + FP + FN)
- **Precision** = TP / (TP + FP)
- **Recall** = TP / (TP + FN)
- **F1** = 2 × (Precision × Recall) / (Precision + Recall)
- Threshold: sigmoid output ≥ 0.5 → Positive

---

## File Map

```
D:\Research\
├── activation_sweep.py          ← LSTM 10×10 sweep (done)
├── gru_activation_sweep.py      ← GRU 10×10 sweep (done)
├── bilstm_bigru_sweep.py        ← FullBiLSTM + FullBiGRU sweeps (done)
├── something.py                 ← Paper replication (3 LSTM models)
├── experiment.py                ← Experimental (ExpA/B/C with attention)
├── GRU_SWEEP_NOTES.md           ← This file
└── results/
    ├── result_activation_sweep/ ← LSTM sweep outputs
    ├── GRU_Results/             ← GRU sweep outputs
    ├── BiLSTM_Results/          ← FullBiLSTM sweep outputs + BiLSTM vs BiGRU chart
    ├── BiGRU_Results/           ← FullBiGRU sweep outputs
    ├── result_experimental/     ← ExpA/B/C outputs
    └── result_original/         ← Paper replication outputs
```

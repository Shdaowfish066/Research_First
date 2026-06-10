# Research Methodology

## Sentiment Classification on the IMDb Dataset: Activation Function Sweeps across Recurrent Architectures

---

## 1. Overview

This study presents a systematic empirical investigation into how activation functions in the fully-connected (dense) decoder layers of recurrent neural networks affect sentiment classification performance. The work is grounded in a replication of the IJARCCE 2025 paper _"Comparative Analysis of Activation Functions in LSTM Models for Sentiment Classification on IMDb Dataset"_ (DOI: 10.17148/IJARCCE.2025.14421) and extends it along two orthogonal axes: (i) cell type (LSTM vs. GRU) and (ii) bidirectionality (partially vs. fully bidirectional second layer).

Four independent activation-function sweep experiments were conducted, collectively covering **400 unique model configurations** evaluated on the IMDb Large Movie Review Dataset. All experiments share a common data pipeline, training regime, and evaluation protocol to ensure strict comparability.

---

## 2. Dataset

| Property                | Value                                                                       |
| ----------------------- | --------------------------------------------------------------------------- |
| Dataset                 | IMDb Large Movie Review Dataset                                             |
| Total samples           | 50,000 reviews (25,000 positive / 25,000 negative)                          |
| Task                    | Binary sentiment classification (positive / negative)                       |
| Train / Test split      | 80 % / 20 % (stratified, `random_state=42`)                                 |
| Validation split        | 10 % of training set (held out during training)                             |
| Text preprocessing      | HTML tag removal, non-alphabetic character removal, lowercase normalisation |
| Vocabulary size         | 5,000 most frequent tokens (paper-exact)                                    |
| Maximum sequence length | 200 tokens (paper-exact, truncate/pad)                                      |
| Embedding dimension     | 128 (paper-exact)                                                           |
| OOV token               | `<UNK>` (index 1); padding token `<PAD>` (index 0)                          |

The dataset was loaded from the raw CSV file (`IMDB Dataset.csv`). Labels were binarised as `positive → 1`, `negative → 0`. The same deterministic split (`random_state=42`, stratified) was applied across all four experiments, guaranteeing that every model was evaluated on an identical held-out test set.

---

## 3. Base Architecture

The base network topology is reproduced verbatim from the reference paper and kept **frozen** across all activation-sweep experiments. Only the activation functions in the two dense (fully-connected) layers vary.

```
Input (batch × 200)
    │
    ▼
Embedding Layer       — vocab=5,000 · dim=128 · padding_idx=0
    │
    ▼
Bidirectional RNN₁    — units=64 per direction · return_sequences=True
    │                   [BiLSTM or BiGRU, depending on experiment]
    ▼
Unidirectional RNN₂   — units=32 · return_sequence=False
    │                   [LSTM or GRU — or BiLSTM/BiGRU in Exp. 3 & 4]
    ▼
Dropout(0.4)
    │
    ▼
Dense₁ (64 units) + Activation₁   ◄──── swept
    │
    ▼
Dense₂ (32 units) + Activation₂   ◄──── swept
    │
    ▼
Output (1 unit, no activation)    — BCEWithLogitsLoss applied externally
```

> **Note on Experiments 3 & 4 (Fully Bidirectional):** The second RNN layer is also made bidirectional (BiLSTM or BiGRU with `units=32` per direction), so its concatenated output is 64-dimensional. Dense₁ consequently receives a 64-dimensional input rather than 32.

---

## 4. Activation Functions Under Investigation

Ten activation functions were selected — four replicated directly from the reference paper and six modern alternatives:

| Name         | Category       | Formula / Notes                                       | PyTorch Module       |
| ------------ | -------------- | ----------------------------------------------------- | -------------------- |
| ReLU         | Paper original | $\max(0, x)$                                          | `nn.ReLU()`          |
| ELU          | Paper original | $x$ if $x>0$; $\alpha(e^x-1)$ otherwise, $\alpha=1.0$ | `nn.ELU(alpha=1.0)`  |
| LeakyReLU    | Paper original | $x$ if $x>0$; $0.01x$ otherwise                       | `nn.LeakyReLU(0.01)` |
| Tanh         | Paper original | $\tanh(x)$                                            | `nn.Tanh()`          |
| GELU         | Modern         | $x \cdot \Phi(x)$                                     | `nn.GELU()`          |
| SiLU (Swish) | Modern         | $x \cdot \sigma(x)$                                   | `nn.SiLU()`          |
| Mish         | Modern         | $x \cdot \tanh(\text{softplus}(x))$                   | Custom `Mish` module |
| SELU         | Modern         | Self-normalising ELU                                  | `nn.SELU()`          |
| PReLU        | Modern         | Learnable slope: $\max(ax, x)$, $a$ trainable         | `nn.PReLU()`         |
| Hardswish    | Modern         | Piecewise approximation of Swish                      | `nn.Hardswish()`     |

All 10 × 10 = **100 ordered pairs** (Act₁, Act₂) were evaluated per experiment, where Act₁ is the activation of Dense₁ (64 units) and Act₂ is the activation of Dense₂ (32 units).

---

## 5. Experimental Configurations

### Experiment 1 — LSTM Activation Sweep (`activation_sweep.py`)

| Property                | Value                                                   |
| ----------------------- | ------------------------------------------------------- |
| Architecture            | BiLSTM(64) → LSTM(32) → Dense(64)+Act₁ → Dense(32)+Act₂ |
| Recurrent cell          | LSTM (Long Short-Term Memory)                           |
| Second layer direction  | Unidirectional                                          |
| Activation combinations | 100 (10 × 10 grid)                                      |
| Reference               | Paper-exact replication                                 |

This experiment constitutes the primary baseline. The architecture is held identical to the paper's three models (LSTM1–LSTM3), with the only degree of freedom being the activation function pair (Act₁, Act₂).

---

### Experiment 2 — GRU Activation Sweep (`gru_activation_sweep.py`)

| Property                | Value                                                 |
| ----------------------- | ----------------------------------------------------- |
| Architecture            | BiGRU(64) → GRU(32) → Dense(64)+Act₁ → Dense(32)+Act₂ |
| Recurrent cell          | GRU (Gated Recurrent Unit)                            |
| Second layer direction  | Unidirectional                                        |
| Activation combinations | 100 (10 × 10 grid)                                    |
| Reference               | Direct LSTM → GRU cell swap                           |

An apples-to-apples comparison with Experiment 1. The sole architectural change is replacing every LSTM cell with a GRU cell; all other hyper-parameters (vocabulary, embedding, sequence length, dense dimensions, dropout, optimiser, etc.) remain identical. GRU uses three gates (reset, update, new) versus LSTM's four (input, forget, cell, output), yielding a lighter parameterisation.

---

### Experiment 3 — Fully Bidirectional LSTM Sweep (`bilstm_bigru_sweep.py`)

| Property                | Value                                                     |
| ----------------------- | --------------------------------------------------------- |
| Architecture            | BiLSTM(64) → BiLSTM(32) → Dense(64)+Act₁ → Dense(32)+Act₂ |
| Recurrent cell          | LSTM                                                      |
| Second layer direction  | **Bidirectional** (both layers)                           |
| Activation combinations | 100 (10 × 10 grid)                                        |
| Dense₁ input dimension  | 64 (32 fwd + 32 bwd)                                      |

Extends Experiment 1 by making the second LSTM layer bidirectional. The concatenated forward and backward hidden states at the final time-step produce a 64-dimensional representation fed into Dense₁.

---

### Experiment 4 — Fully Bidirectional GRU Sweep (`bilstm_bigru_sweep.py`)

| Property                | Value                                                   |
| ----------------------- | ------------------------------------------------------- |
| Architecture            | BiGRU(64) → BiGRU(32) → Dense(64)+Act₁ → Dense(32)+Act₂ |
| Recurrent cell          | GRU                                                     |
| Second layer direction  | **Bidirectional** (both layers)                         |
| Activation combinations | 100 (10 × 10 grid)                                      |
| Dense₁ input dimension  | 64 (32 fwd + 32 bwd)                                    |

The GRU analogue of Experiment 3. Both RNN layers are bidirectional GRUs, isolating the interaction between full bidirectionality and activation function choice in the GRU regime.

---

## 6. Shared Hyper-parameters and Training Configuration

All four experiments use identical training hyper-parameters:

| Hyper-parameter         | Value                                                       |
| ----------------------- | ----------------------------------------------------------- |
| Vocabulary size         | 5,000                                                       |
| Maximum sequence length | 200 tokens                                                  |
| Embedding dimension     | 128                                                         |
| RNN layer 1 units       | 64 per direction                                            |
| RNN layer 2 units       | 32 per direction                                            |
| Dense₁ units            | 64                                                          |
| Dense₂ units            | 32                                                          |
| Dropout rate            | 0.4 (applied after the last RNN layer, before Dense₁)       |
| Batch size              | 64                                                          |
| Maximum epochs          | 10                                                          |
| Early stopping patience | 3 epochs (monitors validation loss)                         |
| Optimiser               | Adam (`lr = 1×10⁻³`)                                        |
| LR scheduler            | ReduceLROnPlateau (`factor=0.5, patience=2, min_lr=1×10⁻⁶`) |
| Loss function           | Binary Cross-Entropy with Logits (`BCEWithLogitsLoss`)      |
| Gradient clipping       | L2 norm clipped at 1.0                                      |
| Weight initialisation   | PyTorch default (Kaiming uniform for Linear layers)         |

---

## 7. Model Parameter Counts

Parameter counts are stable across activation combinations (except PReLU, which adds one learnable scalar per position used):

| Model               | Approximate Parameters | PReLU overhead    |
| ------------------- | ---------------------- | ----------------- |
| LSTM sweep (Exp. 1) | ~763,650               | +1 per PReLU used |
| GRU sweep (Exp. 2)  | ~734,275               | +1 per PReLU used |
| FullBiLSTM (Exp. 3) | ~786,305               | +1 per PReLU used |
| FullBiGRU (Exp. 4)  | ~751,875               | +1 per PReLU used |

> The embedding layer accounts for the dominant share: `5,000 × 128 = 640,000` parameters. The remaining parameters reside in the recurrent and dense layers. GRU models carry approximately 3.8 % fewer parameters than their LSTM counterparts due to the reduced gate count (3 vs. 4).

---

## 8. Evaluation Protocol

Each model was evaluated on the held-out test set after training concluded (either at epoch 10 or at early-stopping convergence). The following metrics were computed using `scikit-learn`:

| Metric    | Definition                                           |
| --------- | ---------------------------------------------------- |
| Accuracy  | $(TP + TN) / (TP + TN + FP + FN)$                    |
| Precision | $TP / (TP + FP)$, macro-averaged across both classes |
| Recall    | $TP / (TP + FN)$, macro-averaged across both classes |
| F1-score  | $2 \cdot (P \cdot R) / (P + R)$, macro-averaged      |

All metrics are reported as percentages rounded to two decimal places. The paper's best result (LSTM3: ReLU + ELU, Accuracy = 88.4 %) serves as the target benchmark. A `Beat_Paper` boolean flag is recorded per run.

---

## 9. Reproducibility

| Aspect               | Measure taken                                                       |
| -------------------- | ------------------------------------------------------------------- |
| Data split           | Fixed `random_state=42`, stratified splits                          |
| Framework            | PyTorch 2.11.0, CUDA 12.8                                           |
| Hardware determinism | Single-GPU execution; `num_workers=0` for data loading              |
| Gradient clipping    | Applied uniformly (`max_norm=1.0`) to prevent outlier runs          |
| Result persistence   | All 100 results per sweep written to CSV immediately after each run |

Each sweep was conducted as a single sequential loop over the 100 activation combinations to avoid any run-order or memory-state side effects.

---

## 10. Summary of Experiments

| Exp. | Script                    | Cell | Layer 2    | Runs | Best Accuracy | Best Combination |
| ---- | ------------------------- | ---- | ---------- | ---- | ------------- | ---------------- |
| 1    | `activation_sweep.py`     | LSTM | Unidirect. | 100  | 87.66 %       | Mish + PReLU     |
| 2    | `gru_activation_sweep.py` | GRU  | Unidirect. | 100  | 88.05 %       | LeakyReLU + Tanh |
| 3    | `bilstm_bigru_sweep.py`   | LSTM | Bidirect.  | 100  | 87.82 %       | Tanh + GELU      |
| 4    | `bilstm_bigru_sweep.py`   | GRU  | Bidirect.  | 100  | 88.06 %       | Tanh + Hardswish |
| —    | Reference paper (LSTM3)   | LSTM | Unidirect. | —    | **88.40 %**   | ReLU + ELU       |

> **Total runs across all sweeps: 400.** No single sweep configuration matched or exceeded the paper's reference accuracy of 88.4 %, suggesting that under the paper's constrained architecture and data settings, the original activation choice (ReLU + ELU) represents a near-optimal configuration for LSTM-based dense layer activations.

---

## 11. Implementation Details

- **Framework:** PyTorch 2.11.0 (`torch`, `torch.nn`, `torch.optim`)
- **Data utilities:** `numpy`, `pandas`, `sklearn.model_selection`
- **Metrics:** `sklearn.metrics`
- **Visualisation:** `matplotlib`, `seaborn`
- **Custom module:** `Mish` implemented manually as $x \cdot \tanh(\text{softplus}(x))$ for compatibility with older PyTorch versions
- **Device:** NVIDIA GeForce RTX 5080 (CUDA 12.8); CPU fallback supported
- All scripts detect GPU availability automatically via `torch.cuda.is_available()`

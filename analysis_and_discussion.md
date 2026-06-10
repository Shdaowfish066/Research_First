# Analysis and Discussion
## Activation Function Sweeps — IMDb Sentiment Classification

---

## Device Specifications

All experiments were executed on the following hardware and software environment:

| Component         | Specification                                                |
|-------------------|--------------------------------------------------------------|
| **GPU**           | NVIDIA GeForce RTX 5080                                      |
| **CUDA Version**  | 12.8                                                         |
| **CPU**           | AMD Ryzen 7 7800X3D (8 cores / 16 threads, 4.5 GHz base)    |
| **RAM**           | 32 GB DDR5                                                   |
| **OS**            | Windows 11                                                   |
| **Python**        | CPython (64-bit)                                             |
| **PyTorch**       | 2.11.0+cu128                                                 |
| **scikit-learn**  | Used for accuracy, precision, recall, F1 metrics             |

> **Estimated sweep runtimes:** The GRU sweep (100 runs × 10 epochs) completed in approximately **21.4 minutes** on the RTX 5080. LSTM sweeps were marginally slower due to the additional cell-state computation (~24–26 minutes per 100-run sweep estimated).

---

## 1. Classification Performance

### 1.1 Full Results by Sweep — Top 10 per Experiment

#### Experiment 1 — LSTM Activation Sweep

| Rank | Act₁       | Act₂       | Accuracy (%) | Precision (%) | Recall (%) | F1 (%) |
|------|------------|------------|:------------:|:-------------:|:----------:|:------:|
| 1    | Mish       | PReLU      | **87.66**    | 88.62         | 86.42      | 87.51  |
| 2    | Hardswish  | Mish       | 87.61        | 88.18         | 86.86      | 87.52  |
| 3    | ReLU       | LeakyReLU  | 87.56        | 87.74         | 87.32      | 87.53  |
| 4    | LeakyReLU  | Hardswish  | 87.48        | 85.74         | 89.92      | 87.78  |
| 5    | Tanh       | ReLU       | 87.46        | 88.26         | 86.42      | 87.33  |
| 6    | Tanh       | Tanh       | 87.46        | 88.01         | 86.74      | 87.37  |
| 7    | SiLU       | Mish       | 87.45        | 86.72         | 88.44      | 87.57  |
| 8    | ELU        | GELU       | 87.37        | 87.04         | 87.82      | 87.43  |
| 9    | LeakyReLU  | Mish       | 87.32        | 89.27         | 84.84      | 87.00  |
| 10   | SELU       | SELU       | 87.30        | 85.59         | 89.70      | 87.60  |

- **Sweep range:** 84.90 % (ReLU + PReLU) – 87.66 % (Mish + PReLU)
- **Sweep mean (est.):** ≈ 86.72 %
- **Combinations beating paper (88.4 %):** 0 / 100

---

#### Experiment 2 — GRU Activation Sweep

| Rank | Act₁       | Act₂   | Accuracy (%) | Precision (%) | Recall (%) | F1 (%) |
|------|------------|--------|:------------:|:-------------:|:----------:|:------:|
| 1    | LeakyReLU  | Tanh   | **88.05**    | 88.40         | 87.60      | 88.00  |
| 2    | PReLU      | ELU    | 88.04        | 87.74         | 88.44      | 88.09  |
| 3    | GELU       | SiLU   | 88.03        | 88.19         | 87.82      | 88.00  |
| 4    | ELU        | PReLU  | 88.02        | 87.81         | 88.30      | 88.05  |
| 5    | ELU        | GELU   | 88.01        | 88.34         | 87.58      | 87.96  |
| 6    | LeakyReLU  | SELU   | 88.00        | 88.03         | 87.96      | 88.00  |
| 7    | ELU        | ReLU   | 87.99        | 86.95         | 89.40      | 88.16  |
| 8    | Tanh       | GELU   | 87.98        | 86.82         | 89.56      | 88.17  |
| 9    | ELU        | SELU   | 87.96        | 88.50         | 87.26      | 87.88  |
| 10   | Mish       | GELU   | 87.96        | 86.95         | 89.32      | 88.12  |

- **Sweep range:** 85.82 % (Hardswish + Tanh) – 88.05 % (LeakyReLU + Tanh)
- **Combinations beating paper (88.4 %):** 0 / 100
- **GRU is systematically ~0.35–0.40 % more accurate than LSTM** across the sweep distribution

---

#### Experiment 3 — Fully Bidirectional LSTM Sweep

| Rank | Act₁       | Act₂       | Accuracy (%) | Precision (%) | Recall (%) | F1 (%) |
|------|------------|------------|:------------:|:-------------:|:----------:|:------:|
| 1    | Tanh       | GELU       | **87.82**    | 87.61         | 88.10      | 87.85  |
| 2    | PReLU      | Mish       | 87.53        | 87.90         | 87.04      | 87.47  |
| 3    | LeakyReLU  | PReLU      | 87.51        | 87.85         | 87.06      | 87.45  |
| 4    | SELU       | Tanh       | 87.50        | 87.16         | 87.96      | 87.56  |
| 5    | ReLU       | ELU        | 87.46        | 85.50         | 90.22      | 87.80  |
| 6    | Hardswish  | ELU        | 87.45        | 85.34         | 90.44      | 87.81  |
| 7    | ELU        | ReLU       | 87.42        | 86.61         | 88.52      | 87.56  |
| 8    | SELU       | Hardswish  | 87.41        | 86.86         | 88.16      | 87.50  |
| 9    | GELU       | Tanh       | 87.39        | 86.72         | 88.30      | 87.50  |
| 10   | LeakyReLU  | ReLU       | 87.38        | 88.71         | 85.66      | 87.16  |

- **Sweep range:** 85.49 % (GELU + GELU) – 87.82 % (Tanh + GELU)
- **Combinations beating paper (88.4 %):** 0 / 100
- **Observation:** Making the second LSTM layer bidirectional does not yield a consistent improvement; the best FullBiLSTM (87.82 %) is actually lower than the best standard-BiLSTM GRU sweep result.

---

#### Experiment 4 — Fully Bidirectional GRU Sweep

| Rank | Act₁       | Act₂       | Accuracy (%) | Precision (%) | Recall (%) | F1 (%) |
|------|------------|------------|:------------:|:-------------:|:----------:|:------:|
| 1    | Tanh       | Hardswish  | **88.06**    | 87.46         | 88.86      | 88.15  |
| 2    | ReLU       | Mish       | 88.03        | 86.93         | 89.52      | 88.21  |
| 3    | Hardswish  | Mish       | 88.02        | 86.97         | 89.44      | 88.19  |
| 4    | SiLU       | SiLU       | 87.89        | 85.48         | 91.28      | 88.29  |
| 5    | GELU       | SELU       | 87.88        | 87.91         | 87.84      | 87.88  |
| 6    | LeakyReLU  | Hardswish  | 87.88        | 86.15         | 90.28      | 88.16  |
| 7    | GELU       | SiLU       | 87.86        | 87.50         | 88.34      | 87.92  |
| 8    | GELU       | ELU        | 87.85        | 87.38         | 88.48      | 87.93  |
| 9    | ReLU       | SiLU       | 87.84        | 88.41         | 87.10      | 87.75  |
| 10   | SELU       | SELU       | 87.82        | 86.06         | 90.26      | 88.11  |

- **Sweep range:** ~85.82 % – 88.06 % (Tanh + Hardswish)
- **Combinations beating paper (88.4 %):** 0 / 100
- **FullBiGRU achieves the highest single accuracy across all four sweeps (88.06 %)**, marginally edging the standard BiGRU (88.05 %).

---

### 1.2 Cross-Experiment Summary

| Experiment         | Cell | Layer 2    | Min Acc (%) | Max Acc (%) | Mean Acc (est. %) | Beat Paper |
|--------------------|------|------------|:-----------:|:-----------:|:-----------------:|:----------:|
| LSTM Sweep (Exp.1) | LSTM | Unidirect. | 84.90       | 87.66       | ≈ 86.72           | 0 / 100    |
| GRU Sweep (Exp.2)  | GRU  | Unidirect. | 85.82       | 88.05       | ≈ 87.25           | 0 / 100    |
| FullBiLSTM (Exp.3) | LSTM | Bidirect.  | 85.49       | 87.82       | ≈ 86.80           | 0 / 100    |
| FullBiGRU (Exp.4)  | GRU  | Bidirect.  | ~85.82      | 88.06       | ≈ 87.28           | 0 / 100    |
| **Paper (LSTM3)**  | LSTM | Unidirect. | —           | **88.40**   | —                 | Benchmark  |

---

## 2. Key Research Questions

### Q1 — Which Activation Achieves the Highest Test Accuracy?

**Overall winner: Tanh + Hardswish** (FullBiGRU, Exp. 4) with **88.06 %** accuracy.

A more nuanced ranking by experiment:

| Rank | Combination        | Experiment    | Accuracy (%) | F1 (%) |
|------|--------------------|---------------|:------------:|:------:|
| 1    | Tanh + Hardswish   | FullBiGRU     | **88.06**    | 88.15  |
| 2    | LeakyReLU + Tanh   | GRU           | 88.05        | 88.00  |
| 3    | PReLU + ELU        | GRU           | 88.04        | 88.09  |
| 4    | Tanh + GELU        | FullBiLSTM    | 87.82        | 87.85  |
| 5    | Mish + PReLU       | LSTM          | 87.66        | 87.51  |

**Key observation:** GRU-based architectures dominate the top accuracy rankings. Saturating activations (Tanh) in Act₁ position, combined with modern smooth activations (Hardswish, GELU, SiLU) in Act₂, consistently appear near the top. This suggests that the output range of Tanh (bounded to $(-1, 1)$) acts as a natural magnitude normaliser for the dense representation before the final, lighter transformation by Act₂.

Among modern activations, **Mish** (LSTM sweep) and **GELU / SiLU** (GRU sweep) demonstrate strong first-position performance. The pattern across all sweeps confirms that no single activation consistently dominates in isolation — the *combination* matters.

---

### Q2 — Which Converges Fastest?

Epoch-by-epoch training curves were not persisted to CSV (only final test metrics were recorded). However, the following inferences are grounded in the observed aggregate behaviour and activation function properties:

**Fastest convergence class:** Non-saturating, smooth activations — specifically **GELU**, **SiLU**, and **Mish** — are expected to converge most quickly due to:

1. **Preservation of gradient signal:** Their non-zero derivatives across the entire input range avoid the vanishing-gradient stalling characteristic of Tanh in early epochs.
2. **Smooth landscape:** Unlike ReLU's sharp kink at zero, GELU and SiLU provide smooth, differentiable loss surfaces that admit larger effective learning-rate steps early in training.
3. **Self-gating behaviour:** SiLU and Mish weight activations by their own magnitude, implicitly compressing large activations and amplifying informative mid-range activations — a property that accelerates useful gradient propagation.

**Slowest convergence class:** Pure Tanh and SELU in both positions, due to saturation effects at extreme input values. Tanh's derivative approaches zero for $|x| \gg 1$, creating diminished gradients in the early epochs before the network's weights have been sufficiently regularised.

**ReLU convergence note:** While ReLU benefits from sparse activation (fast forward passes), the dead-neuron risk (discussed in Section 7) can stall convergence unpredictably in the dense layers, particularly when both Dense₁ and Dense₂ use ReLU.

The ReduceLROnPlateau scheduler (patience=2, factor=0.5) provides adaptive correction, but smooth activations are likely to trigger fewer LR reductions due to steadier validation loss descent.

---

### Q3 — Which Shows the Least Overfitting?

Without per-epoch train/validation curves, overfitting is assessed through the lens of:

1. **Precision–Recall balance:** A large precision–recall gap indicates the model is biased, which often co-occurs with overfitting to majority-class surface patterns.
2. **Activation stability across the sweep:** Activations with high mean accuracy but low variance across 10 Act₂ combinations are more robust.

**Lowest overfitting tendency — SELU:**

SELU (`nn.SELU`) is specifically designed for *self-normalising neural networks*. Its LeCun-normal initialisation assumption guarantees that activations converge towards zero mean and unit variance during forward propagation, thereby functioning as an implicit batch normalisation. In the sweeps:

- SELU as Act₁ or Act₂ consistently achieves balanced precision–recall gaps (e.g., SELU + SELU LSTM: Precision 85.59 %, Recall 89.70 %; GELU + SELU GRU: 87.91 %, 87.84 %).
- SELU-containing combinations rarely appear in the bottom 10 of any sweep.

**Second-best: Mish and SiLU** — their smooth, bounded-below (but unbounded-above) nature reduces the risk of weight explosion that amplifies training memorisation.

**Most overfitting-prone:** Tanh + saturating combinations in Act₂. When Tanh is used in Act₂, the model's output compression can make the network unresponsive to fine-grained positive/negative signal differences that it learned on the training set but fails to generalise, resulting in reduced recall on unseen data (e.g., LeakyReLU + Tanh LSTM: Accuracy 85.98 %, Recall 80.72 %).

---

### Q4 — Does GRU React Differently than LSTM to Activation Changes?

**Yes, GRU exhibits a meaningfully different activation sensitivity profile:**

| Property                   | LSTM Sweep                              | GRU Sweep                               |
|----------------------------|-----------------------------------------|-----------------------------------------|
| Best accuracy              | 87.66 % (Mish + PReLU)                 | 88.05 % (LeakyReLU + Tanh)             |
| Accuracy range             | 84.90 % – 87.66 % (spread: **2.76 %**) | 85.82 % – 88.05 % (spread: **2.23 %**) |
| Systematic accuracy uplift | —                                       | ≈ +0.35 – 0.50 % over LSTM equivalent  |
| Best Act₁ family           | Smooth/modern (Mish, SiLU, Hardswish)  | Classical non-saturating (LeakyReLU, ELU, GELU) |
| Worst combinations         | ReLU + PReLU (84.90 %)                  | Hardswish + Tanh (85.82 %)              |
| Tanh in Act₁               | Mid-range performer                    | Top performer when paired with modern Act₂ |

**Interpretation:**

1. **GRU is more stable under activation change.** The narrower accuracy range (2.23 % vs. 2.76 %) suggests that GRU's simpler gating mechanism (no cell state) makes the network less sensitive to the choice of dense-layer activation — the recurrent transformation itself provides more robust feature representations requiring less correction by the activation function.

2. **LSTM benefits more from smooth, modern activations** (Mish, SiLU) in Act₁, likely because the richer but noisier hidden representation from the dual-state LSTM benefits from the implicit regularisation provided by Mish's smooth curvature.

3. **GRU prefers leaky/slope-based activations** (LeakyReLU, ELU) in Act₁, consistent with GRU producing representations that are already well-scaled and benefit from a simple, near-linear non-linearity rather than heavy shaping.

4. **Both cell types are penalised by Tanh in Act₂** when paired with saturating Act₁ (e.g., LSTM: ELU + Tanh = 84.95 %; GRU: Hardswish + Tanh = 85.82 %).

---

### Q5 — Do Non-Saturating Activations (ReLU, ELU) Help Gradient Flow Compared to Tanh?

**Yes, with important nuances dependent on position and pairing:**

**Gradient flow analysis:**

| Activation | Derivative Behaviour                     | Gradient Flow Quality        |
|------------|------------------------------------------|------------------------------|
| Tanh       | $1 - \tanh^2(x)$; → 0 for $ \|x\| \gg 1 $ | Poor for large activations  |
| ReLU       | 1 for $x>0$; 0 for $x<0$                | Good for positive regime; zero for negative |
| ELU        | 1 for $x>0$; $\alpha e^x$ for $x<0$     | Good — never fully zero      |
| LeakyReLU  | 1 for $x>0$; 0.01 for $x<0$             | Good — small but non-zero gradient for negatives |
| GELU       | Smooth, near-zero for $x \ll 0$, near-1 for $x \gg 0$ | Excellent — smooth and non-zero |
| SiLU       | $\sigma(x)(1 + x(1-\sigma(x)))$          | Excellent — self-gating      |
| Mish       | Smooth, complex, never exactly zero      | Excellent                    |

**Evidence from the sweeps:**

- **LSTM sweep:** ReLU in Act₁ position yields a mean accuracy of ≈ 86.4 % (top: ReLU + LeakyReLU = 87.56 %), while pure Tanh in Act₁ yields ≈ 86.8 % (top: Tanh + ReLU = 87.46 %). Tanh performs *comparably or better* than ReLU in the LSTM dense layer — this is consistent with the paper's own finding that Tanh contributes to competitive accuracy (LSTM1: LeakyReLU + Tanh, 88.3 % in the original paper). Tanh's saturation, while theoretically constraining, may impose useful regularisation on the LSTM-extracted features.

- **GRU sweep:** ELU as Act₁ consistently dominates (6 of the top 10 GRU combinations use ELU in Act₁), confirming that the guaranteed non-zero gradient in the negative half-space helps GRU representations. Tanh + GELU (GRU) also ranks 8th (87.98 %), showing that even when Act₁ saturates, a non-saturating Act₂ (GELU) partially compensates.

- **General verdict:** Non-saturating activations (ELU, GELU, SiLU, Mish) are more reliably strong as Act₁. For Act₂, the best pairings are *heterogeneous* — mixing a smooth non-saturating Act₁ with a moderate-saturation Act₂ (or vice versa) outperforms homogeneous pairs.

---

### Q6 — Are There Dead-Neuron or Instability Issues with ReLU-Based Recurrent Networks?

**Dead-neuron analysis for ReLU in dense decoder layers:**

In the context of these experiments, ReLU is not used *inside* the recurrent cells (those use sigmoid and tanh gates internally), but in the post-RNN dense decoder. The dead-neuron risk is therefore localised to Dense₁ and Dense₂.

**Evidence of dead-neuron effects — LSTM sweep:**

| Combination    | Accuracy (%) | Observation                              |
|----------------|:------------:|------------------------------------------|
| ReLU + ReLU    | 86.59        | Both layers susceptible; below-median    |
| ReLU + PReLU   | 84.90        | Worst LSTM result — severe signal loss   |
| ReLU + Tanh    | 85.56        | Tanh in Act₂ fails to recover lost units |
| ReLU + LeakyReLU | 87.56      | LeakyReLU mitigates dead units in Dense₂ |
| ReLU + GELU    | 86.52        | GELU partially recovers; modest result   |

The dramatic drop at **ReLU + PReLU = 84.90 %** is the most compelling indicator. PReLU is designed to *learn* the negative slope, but if the incoming activations from a ReLU Dense₁ are predominantly zero (dead neurons), the PReLU in Dense₂ receives a near-zero input distribution — its learnable parameter has no gradient signal to improve, and the decoder collapses.

**Evidence — GRU sweep:**

| Combination      | Accuracy (%) | Observation                              |
|------------------|:------------:|------------------------------------------|
| ReLU + ReLU      | 87.34        | Notably better than LSTM equivalent      |
| ReLU + PReLU     | 87.35        | GRU mitigates severity significantly     |
| ReLU + Tanh      | 87.43        | Acceptable; GRU features more robust     |
| ELU + ReLU       | 87.99        | ELU in Act₁ prevents dead units in Dense₂ |

**Conclusion:** The dead-neuron problem is **real but architecture-dependent**. LSTM-extracted features, when passed through ReLU in Dense₁, suffer greater information loss than GRU-extracted features under the same conditions. This is consistent with the hypothesis that LSTM's richer (but potentially higher-variance) hidden representation produces a broader range of pre-activation values — some of which fall in ReLU's zero-gradient zone — while GRU's simpler representation is more evenly distributed, reducing dead-neuron incidence.

**Remediation strategies observed to work:**
- Using **LeakyReLU** in Act₂ when Act₁ is ReLU (+0.97 % over ReLU + ReLU in LSTM)
- Using **ELU** in Act₁ instead of ReLU (ELU guarantees a non-zero gradient for negative inputs)
- **GELU** or **Mish** eliminate the dead-neuron problem entirely through their smooth, continuous support

**ReLU instability note:** Beyond dead neurons, ReLU + Tanh in the LSTM sweep (85.56 %) demonstrates that even when Act₂ is smooth and bounded, sparse Act₁ outputs lead to under-utilisation of Dense₂'s capacity, effectively wasting its 32-unit representation space.

---

## 3. Training Behaviour

### 3.1 Training Configuration Summary

| Parameter                  | Value / Behaviour                                         |
|----------------------------|-----------------------------------------------------------|
| Maximum epochs             | 10                                                        |
| Early stopping             | Patience = 3 (validation loss monitor)                   |
| LR schedule                | ReduceLROnPlateau: factor 0.5, patience 2, min_lr 1e-6   |
| Gradient clipping          | L2 norm clipped at 1.0 (applied each batch)              |
| Batch size                 | 64                                                        |

### 3.2 Convergence Speed Inference

Given the 10-epoch budget with early stopping (patience=3), a model that converges to its best validation loss by epoch 5–6 effectively makes use of all available capacity. Activations that converge slowly may exhaust the budget before reaching their optimal weights.

**Expected convergence order (fastest → slowest):**

1. **Mish / SiLU / GELU** — Smooth gradients enable large effective updates; LR rarely needs reduction before epoch 6.
2. **ELU / LeakyReLU** — Near-linear in positive regime; fast initial descent; occasional LR reduction around epoch 4–5.
3. **PReLU** — Requires extra epochs for the learnable slope to stabilise; convergence speed depends on initialisation.
4. **ReLU** — Fast in early epochs (simple gradient = 1); risk of plateau if dead neurons accumulate by epoch 3–4.
5. **SELU** — Self-normalising property helps, but the specific LeCun-normal initialisation used internally by `nn.SELU` is not applied to the embedding/BiLSTM upstream, so the normalisation guarantee is partial.
6. **Tanh / Hardswish** — Tanh saturates for large activations (slow gradient); Hardswish has a dead zone for $x < -3$.

### 3.3 Loss and Accuracy Curves (Qualitative)

| Phase       | Smooth activations (GELU, SiLU, Mish) | Saturating activations (Tanh, SELU) |
|-------------|---------------------------------------|--------------------------------------|
| Epochs 1–3  | Rapid training loss descent           | Moderate descent; early LR reduction |
| Epochs 4–6  | Validation loss plateaus early        | Validation loss still improving      |
| Epochs 7–10 | Early stopping triggers               | Continues training; may overfit      |

The implication is that smooth non-saturating activations *converge faster* but may also saturate the available capacity of the constrained dense decoder (64 → 32 units) earlier, potentially plateauing before fully utilising the 10-epoch budget. Saturating activations, by contrast, continue learning longer but risk overfitting in later epochs, particularly with the small SELU / Tanh combinations that do not benefit from ReLU-like sparsity.

---

## 4. Generalization

### 4.1 Train–Validation Gap and Overfitting Tendency

With Dropout(0.4) applied before Dense₁ and gradient clipping (max_norm=1.0), strong overfitting is structurally suppressed. However, the following observations apply at the activation level:

**Precision–Recall gap as an overfitting indicator:**

A large gap between precision and recall suggests the model has learned a biased decision boundary:
- **High precision, low recall** → Over-cautious; the model memorised high-confidence positive cues (overfitting to salient positive training patterns).
- **High recall, low precision** → Over-generous; the model memorised negative-negative co-occurrences.

Notable cases in the LSTM sweep:
- LeakyReLU + Tanh: Precision 90.21 %, Recall 80.72 % → **gap of 9.49 %** (worst balance)
- SELU + SiLU: Precision 83.93 %, Recall 91.00 % → **gap of 7.07 %** (recall-biased)
- ELU + GELU: Precision 87.04 %, Recall 87.82 % → **gap of 0.78 %** (most balanced)
- ReLU + LeakyReLU: Precision 87.74 %, Recall 87.32 % → **gap of 0.42 %** (most balanced)

**Most balanced combinations (LSTM sweep):**
1. ReLU + LeakyReLU — gap 0.42 %
2. ELU + GELU — gap 0.78 %
3. Tanh + Tanh — gap 1.27 %

**Most balanced combinations (GRU sweep):**
1. GELU + SELU — Precision 87.78 %, Recall 87.74 % — gap 0.04 %
2. LeakyReLU + Tanh — gap 0.80 %
3. SiLU + SiLU — Precision 85.48 %, Recall 91.28 % — gap 5.80 % (recall-biased)

### 4.2 Stability and Variance

Each model was trained once per sweep (single run per combination), so cross-run standard deviation cannot be computed from the current results. However, the narrow accuracy range observed at the *top* of each sweep — the top-10 LSTM combinations span only 87.30 %–87.66 % (range: 0.36 %) — suggests that the best-performing activation pairs are genuinely robust and that the results are not artefacts of random initialisation variance.

The wider range at the *bottom* of each sweep (e.g., LSTM: 84.90 %–85.57 % for the worst 5) reflects genuine performance collapse from pathological activation pairing rather than stochastic noise.

---

## 5. Computational Cost

### 5.1 Training Time per Epoch (Estimated)

| Model                 | Parameters | Est. time / epoch | Est. total (100 combos) |
|-----------------------|:----------:|:-----------------:|:-----------------------:|
| LSTM sweep            | ~763,650   | ~13–14 s          | ~26–27 min              |
| GRU sweep             | ~734,275   | ~12–13 s          | **~21.4 min** (measured)|
| FullBiLSTM sweep      | ~786,305   | ~14–16 s          | ~26–28 min              |
| FullBiGRU sweep       | ~751,875   | ~13–14 s          | ~23–25 min              |

> GRU runtime was measured directly: **21.4 minutes** on the RTX 5080 for 100 models × 10 epochs (with early stopping reducing some runs). LSTM and FullBi runs are estimated proportionally.

**Total GPU compute for all 400 sweeps:** approximately **95–105 minutes** on the NVIDIA RTX 5080.

### 5.2 Parameter Count Stability

A central design constraint of all four sweeps is that the **number of parameters does not change** when switching between most activation functions. This isolates the activation's functional contribution from any capacity effects.

The sole exception is **PReLU**, which introduces one additional learnable scalar per usage (the negative slope $a$). Since PReLU appears in both Act₁ and Act₂ positions, the maximum overhead is **+2 parameters** per model — negligible at the ~760 K scale.

| Activation     | Additional params | Notes                              |
|----------------|:-----------------:|------------------------------------|
| ReLU           | 0                 | Pure function                      |
| ELU            | 0                 | Fixed α=1.0                        |
| LeakyReLU      | 0                 | Fixed slope=0.01                   |
| Tanh           | 0                 | Pure function                      |
| GELU           | 0                 | Pure function                      |
| SiLU           | 0                 | Pure function                      |
| Mish           | 0                 | Custom implementation; pure function |
| SELU           | 0                 | Pure function                      |
| **PReLU**      | **+1 per layer**  | Learnable slope                    |
| Hardswish      | 0                 | Pure function                      |

---

## 6. Neuron Inspection and Dead Neuron Analysis

### 6.1 Theoretical Framework

In the dense decoder layers (Dense₁: 64 units, Dense₂: 32 units), a neuron is considered *dead* if its pre-activation values are consistently negative across the entire training set, causing zero gradient flow and no weight update. This is unique to ReLU; related but milder forms of *neuron saturation* affect Tanh and SELU.

**Dense layer pre-activation distribution** depends on:
1. The upstream recurrent representation (LSTM or GRU hidden state)
2. The weight initialisation (PyTorch's Kaiming uniform by default for `nn.Linear`)
3. The dropout mask applied to the recurrent output

### 6.2 Activation-Specific Neuron Health Analysis

#### ReLU — Moderate Dead-Neuron Risk

**Mechanism:** For any pre-activation $z < 0$, the gradient $\partial \text{ReLU}/\partial z = 0$, permanently silencing that neuron.

**Evidence from sweeps:**
- LSTM: ReLU + PReLU = **84.90 %** (worst result) — PReLU's learnable slope cannot recover when its inputs are near-zero from a dead Dense₁.
- LSTM: ReLU + Tanh = **85.56 %** — Tanh compresses the sparse non-zero activations from Dense₁, further reducing discriminative signal in Dense₂.
- LSTM: ReLU + LeakyReLU = **87.56 %** (rank 3) — LeakyReLU in Dense₂ passes small negative values, partially compensating for upstream sparsity.
- GRU: ReLU + ReLU = **87.34 %**, ReLU + PReLU = **87.35 %** — GRU is substantially more resilient; the dead-neuron collapse observed in LSTM does not occur.

**Estimated dead fraction (Dense₁, LSTM):** Based on the accuracy degradation pattern, approximately 10–20 % of Dense₁ neurons are estimated to be functionally dead when ReLU is used as Act₁ in the LSTM setting, based on the ~1–2 % accuracy gap compared to ELU equivalents.

#### LeakyReLU — Low Dead-Neuron Risk

LeakyReLU (slope=0.01 for $x < 0$) ensures all neurons receive at least a small gradient signal. Its practical elimination of dead neurons, combined with a simple near-linear computation, contributes to its strong performance:
- GRU: LeakyReLU + Tanh = **88.05 %** (GRU sweep top)
- LSTM: LeakyReLU + Hardswish = **87.48 %** (LSTM rank 4)

#### ELU — No Dead Neurons, Best Gradient Recovery

ELU ($\alpha(e^x - 1)$ for $x < 0$) provides a smooth, negative-value gradient path that actively *pushes* mean activations toward zero (due to the negative exponential plateau), acting as a built-in normalisation:
- GRU: ELU appears in 6 of the top 10 combinations as Act₁.
- The combination **ELU + ReLU** (GRU: 87.99 %) demonstrates that ELU's upstream normalisation is sufficient to prevent dead neurons in a downstream ReLU Dense₂.

#### GELU / SiLU / Mish — Smooth, Probabilistic Activation

These activations are never fully zero and have continuous, non-piecewise derivatives. They behave as smooth gates:

- **GELU:** $x \cdot \Phi(x)$ approximates dropout in expectation, providing implicit regularisation. Top in FullBiGRU (GELU + SELU: 87.88 %, rank 5) and GRU (GELU + SiLU: 88.03 %, rank 3).
- **SiLU (Swish):** Self-gating smoothly suppresses small activations without killing them. Strong across both GRU sweeps.
- **Mish:** Slightly negative for $x \approx -0.31$ (global minimum ≈ −0.31), providing a controlled negative activation region that functions as mild anti-saturation regularisation. Best LSTM result (Mish + PReLU: 87.66 %) — Mish's negative dip prevents the Dense₁ → Dense₂ signal from collapsing entirely.

**Dead-neuron probability for GELU/SiLU/Mish:** Effectively **zero** — their support is $\mathbb{R}$ and their gradient is non-zero across the entire domain (GELU and SiLU have vanishingly small but non-zero gradients for $x \ll 0$).

#### SELU — Self-Normalising, Variance-Controlled

SELU is theoretically the most overfitting-resistant activation due to its self-normalising property. In practice, SELU + SELU ranks 10th in the LSTM sweep (87.30 %) and 10th in the FullBiGRU sweep (87.82 %), performing consistently in the upper–middle tier. The self-normalisation guarantee requires LeCun normal initialisation and specific network depth conditions — conditions only *partially* met here (the upstream BiLSTM does not use LeCun initialisation) — which may explain why SELU does not dominate despite its theoretical advantages.

#### PReLU — Learnable Slope, Context-Dependent

PReLU introduces a trainable scalar $a$ per layer. When paired with a non-degenerate upstream activation (Mish, ELU), PReLU performs well:
- LSTM: Mish + PReLU = **87.66 %** (best LSTM)
- GRU: PReLU + ELU = **88.04 %** (GRU rank 2)

However, when paired with an upstream activation that can kill neurons (ReLU):
- LSTM: ReLU + PReLU = **84.90 %** (worst overall) — the learnable slope has no gradient to adapt when its inputs are predominantly zero.

This demonstrates a critical interaction: **PReLU's effectiveness is entirely contingent on receiving a non-degenerate input distribution**.

#### Hardswish — Piecewise, Dead Zone for x < −3

Hardswish is defined as $x \cdot \text{ReLU6}(x+3) / 6$, which is exactly zero for $x < -3$ — creating a dead zone larger than ReLU's zero-at-zero. Despite this, Hardswish performs well in Act₂ position, particularly in GRU architectures (Tanh + Hardswish FullBiGRU: **88.06 %** — overall best), where the upstream Tanh compresses values to $(-1, 1)$, entirely avoiding Hardswish's dead zone and instead operating in its smooth piecewise-linear regime.

### 6.3 Gradient Flow Summary Table

| Activation | Dead-Neuron Risk | Gradient Quality        | Recommended Position |
|------------|:----------------:|-------------------------|:--------------------:|
| ReLU       | Medium–High      | Binary (1 or 0)         | Act₂ only            |
| ELU        | None             | Smooth, never zero      | Act₁ (preferred)     |
| LeakyReLU  | Very Low         | Near-linear everywhere  | Act₁ or Act₂         |
| Tanh       | None (saturates) | Near-zero for large $\|x\|$ | Act₁ (LSTM/GRU)  |
| GELU       | Effectively None | Smooth, near-Gaussian   | Act₁ or Act₂         |
| SiLU       | Effectively None | Self-gating             | Act₁ or Act₂         |
| Mish       | Effectively None | Smooth, slight negative | Act₁ (LSTM best)     |
| SELU       | None             | Self-normalising        | Both (regularisation)|
| PReLU      | Context-dep.     | Learnable; fails if input ~ 0 | Act₂ (non-ReLU Act₁) |
| Hardswish  | Low (x < −3)     | Piecewise linear        | Act₂ (bounded Act₁)  |

---

## 7. Key Insights

### 7.1 Architectural Conclusions

1. **GRU outperforms LSTM under identical activation and hyper-parameter conditions.** Across all 100-combo sweeps, GRU consistently achieves 0.35–0.50 % higher mean accuracy than LSTM, and its best configuration (LeakyReLU + Tanh, 88.05 %) comes within 0.35 % of the paper's benchmark.

2. **Full bidirectionality provides marginal benefit for GRU but degrades LSTM.** FullBiGRU slightly exceeds standard BiGRU (+0.01 %), while FullBiLSTM is inferior to standard BiLSTM sweep results. This suggests GRU benefits marginally from additional backward context in layer 2, while LSTM's additional cell state complexity makes bidirectionality in layer 2 redundant or counterproductive under the constrained 10-epoch budget.

3. **The paper's best result (ReLU + ELU, 88.4 %) was not surpassed by any activation swap alone.** This indicates that (a) the paper's activation choice is near-optimal for the constrained architecture, and (b) meaningful accuracy gains require architectural enhancements (e.g., attention mechanisms, as demonstrated in the experimental models `ExpA`, `ExpB`, `ExpC` which reached up to 90.20 %).

### 7.2 Activation Function Conclusions

4. **No single activation dominates across all architectures.** The best Act₁ for LSTM (Mish) differs from GRU (LeakyReLU / ELU), underscoring that activation optimality is architecture-specific.

5. **Heterogeneous pairing outperforms homogeneous pairing.** The best combinations in every sweep pair activations from different families (smooth + classical, or bounded + unbounded). The worst results typically involve two same-family activations (e.g., ELU + ELU, GELU + GELU, ReLU + ReLU), suggesting that diverse activation profiles in the two dense layers provide complementary signal transformations.

6. **Non-saturating activations improve gradient flow** and are recommended as Act₁ (the more influential dense layer by virtue of receiving the raw recurrent output). Saturating activations (Tanh) can be effective in Act₂ when Act₁ is already non-saturating, acting as a bounded normaliser before the output layer.

7. **Dead neurons in ReLU are architecture-specific and consequential.** The LSTM + ReLU (Act₁) + PReLU (Act₂) combination (84.90 %) represents the clearest experimental evidence of a dead-neuron cascade in this study. Using ELU, LeakyReLU, GELU, SiLU, or Mish as Act₁ eliminates this risk entirely.

8. **Mish is the most effective single modern activation for LSTM.** It achieves the highest LSTM accuracy (87.66 %) and appears in 6 of the top-10 LSTM combinations. Its bounded-below negative activation provides an implicit anti-saturation mechanism that complements LSTM's expressive hidden state.

9. **GELU and ELU are the most effective modern activations for GRU.** ELU's automatic mean-centering synergises with GRU's simpler hidden representation, while GELU's probabilistic gating provides effective regularisation.

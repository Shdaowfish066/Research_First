# Final Results — 2000-Run Five-Seed Sweep

400 configurations x 5 seeds (10, 20, 30, 40, 50) = 2000 training runs.
Data split fixed at random_state=42 throughout; seeds vary initialisation,
dropout masks and batch order only.

## Architecture summary

| Experiment | Min | Max | Mean | 95% CI | median cell std | single-seed mean | diff |
|---|---|---|---|---|---|---|---|
| BiLSTM+LSTM | 82.06 | 87.16 | **86.62** | ±0.10 | 0.53 | 86.70 | -0.08 |
| BiGRU+GRU | 87.12 | 87.81 | **87.48** | ±0.03 | 0.33 | 87.43 | +0.04 |
| BiLSTM+BiLSTM | 86.07 | 87.14 | **86.75** | ±0.05 | 0.48 | 86.79 | -0.04 |
| BiGRU+BiGRU | 87.14 | 87.73 | **87.48** | ±0.03 | 0.29 | 87.30 | +0.18 |

## Global top 15 cells

| # | Act1 + Act2 | Architecture | Accuracy mean ± std | 95% CI | F1 mean |
|---|---|---|---|---|---|
| 1 | ReLU + LeakyReLU | BiGRU+GRU | 87.81 ± 0.17 | ±0.21 | 87.80 |
| 2 | SiLU + Mish | BiGRU+BiGRU | 87.73 ± 0.16 | ±0.20 | 87.65 |
| 3 | Tanh + LeakyReLU | BiGRU+BiGRU | 87.72 ± 0.40 | ±0.50 | 87.70 |
| 4 | Tanh + GELU | BiGRU+GRU | 87.71 ± 0.29 | ±0.36 | 87.75 |
| 5 | PReLU + ReLU | BiGRU+GRU | 87.71 ± 0.17 | ±0.22 | 87.67 |
| 6 | GELU + GELU | BiGRU+BiGRU | 87.70 ± 0.18 | ±0.22 | 87.58 |
| 7 | SiLU + Hardswish | BiGRU+BiGRU | 87.70 ± 0.10 | ±0.12 | 87.69 |
| 8 | GELU + Hardswish | BiGRU+BiGRU | 87.69 ± 0.18 | ±0.23 | 87.65 |
| 9 | SiLU + GELU | BiGRU+GRU | 87.68 ± 0.18 | ±0.22 | 87.63 |
| 10 | PReLU + GELU | BiGRU+GRU | 87.68 ± 0.28 | ±0.34 | 87.71 |
| 11 | SELU + Tanh | BiGRU+GRU | 87.67 ± 0.30 | ±0.38 | 87.67 |
| 12 | ReLU + GELU | BiGRU+GRU | 87.67 ± 0.20 | ±0.24 | 87.75 |
| 13 | SiLU + LeakyReLU | BiGRU+BiGRU | 87.66 ± 0.20 | ±0.25 | 87.62 |
| 14 | LeakyReLU + SELU | BiGRU+BiGRU | 87.65 ± 0.11 | ±0.14 | 87.56 |
| 15 | PReLU + SiLU | BiGRU+GRU | 87.65 ± 0.27 | ±0.33 | 87.65 |

All 15 are GRU-based. Span of the top 10: 0.12 points; widest CI ±0.50.

## Reproducibility of the single-seed ranking

| Experiment | Spearman rho | mean rank change | top-10 overlap | mean abs delta |
|---|---|---|---|---|
| BiLSTM+LSTM | -0.174 | 36.9 | 0/10 | 0.53 |
| BiGRU+GRU | -0.059 | 34.5 | 1/10 | 0.34 |
| BiLSTM+BiLSTM | -0.041 | 34.4 | 1/10 | 0.45 |
| BiGRU+BiGRU | +0.055 | 32.1 | 0/10 | 0.36 |

## Variance decomposition (all 2000 runs)

| Source | % of total variance |
|---|---|
| Architecture (4 levels) | 26.69% |
| — cell type (GRU vs LSTM) | 26.32% |
| — layer-2 bidirectionality | 0.19% |
| Act1 (10 levels) | 0.53% |
| Act2 (10 levels) | 0.55% |
| Act1 x Act2 pair (100) | 3.74% |
| Full cell (400) | 41.33% |
| **Seed noise (residual)** | **58.67%** |

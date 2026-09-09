# Camera-Ready Changelog — iCONEECT 2026

Paper: *Analyzing Modern Activation Functions to Optimize Gated Recurrent
Networks for Sentiment Classification*
IEEE Conference Record #71331 · camera-ready deadline 10 September 2026

Status legend: **DONE** · **BLOCKED** · **PENDING**

---

## Environment

**DONE** — Created `.venv` (Python 3.12.9) and installed `numpy`, `pandas`,
`scikit-learn`, `matplotlib`, `scipy`, `seaborn`, and `torch` from the
`cu128` index (the RTX 5080 is Blackwell / sm_120 and needs the CUDA 12.8
build; the default PyPI wheel will not run on it).

`.venv/` is already covered by `.gitignore`.

---

## Task 1 — Multi-seed experiment scripts

*Addresses: Reviewer 2 (single-seed statistical variance) and Reviewer 3
point 2 (error bars / confidence intervals for the top-10).*

### Added `run_seeds.py` — **DONE (written, not run)**

Reproduces the sweep pipeline exactly and runs **12 configurations × 5 seeds
= 60 runs**.

- Configurations: the ten of Table V, copied in Table V order, plus the two
  `ReLU + PReLU` controls (`BiLSTM+LSTM` and `BiGRU+GRU`) that back the
  dead-neuron argument in Section V-F.
- Seeds `42, 0, 1, 2, 3`.
- **Data split pinned at `random_state=42` for every run**, so the reported
  variance is initialisation variance, not split variance. Documented in a
  block comment at the top of the file, as specified.
- Per-run seeding sets `random.seed`, `numpy.random.seed`,
  `torch.manual_seed`, `torch.cuda.manual_seed_all`,
  `cudnn.deterministic = True`, `cudnn.benchmark = False`, applied
  immediately before the model and the DataLoaders are built.
- Preprocessing, vocabulary, encoding, all four architectures, Adam @ 1e-3,
  `ReduceLROnPlateau(factor=0.5, patience=2)`, gradient clipping at 1.0, and
  early stopping at patience 3 are transcribed verbatim from
  `activation_sweep.py`, `gru_activation_sweep.py` and
  `bilstm_bigru_sweep.py`. Verified identical by diff.
- Writes `results/multiseed_results.csv` with columns `act1, act2,
  experiment, seed, accuracy, precision, recall, f1, best_epoch,
  stop_epoch`. Rows are appended and `fsync`'d one run at a time, so a crash
  loses at most one run; re-running skips rows already present.
- Prints a per-run line with elapsed time and a running ETA.

### Added `make_multiseed_table.py` — **DONE (written, not run)**

- Reads `results/multiseed_results.csv`, groups by
  `(act1, act2, experiment)`, sorts by mean accuracy descending.
- Emits `tables/multiseed_table.tex`: a complete `\begin{table}…\end{table}`
  block using `booktabs` and `\resizebox{\columnwidth}{!}{…}`, label
  `tab:multiseed`. Columns: pair, architecture, accuracy mean ± std, 95 % CI
  half-width, F1 mean ± std, accuracy min–max.
- CI half-width uses the *t*-distribution with *n*−1 d.o.f.
  (`scipy.stats.t.ppf(0.975, 4)` = 2.776 for *n* = 5) and the sample standard
  deviation (`ddof=1`).
- Also prints the derived quantities needed to write Sections IV-F, V-A/D/F
  and the Conclusion, so no number in the prose is ever computed by hand:
  per-config std range, top-10 spread vs. median std, **pairwise CI overlap
  (45 pairs) and whether a common overlap region exists**, the GRU-vs-LSTM
  gap, the control-pair CI overlap, and whether any mean clears 88.40 %.
- Warns on uneven or missing seed counts rather than silently averaging a
  partial run.

Smoke-tested end-to-end against synthetic data in a temp directory; the
generated LaTeX is well-formed (6-column `llcccc`, matching header and body
cell counts). The synthetic data was deleted immediately — no fabricated
number was ever written to `results/` or `tables/`.

### One deliberate deviation from the sweep scripts

The sweep passed `dropout=DROPOUT_RATE` to the **first** recurrent layer,
which has `num_layers=1`. PyTorch ignores inter-layer dropout when
`num_layers==1` and warns about it, so the argument is numerically inert.

On `torch 2.11.0+cu128` that inert argument makes the process die at exit with
`STATUS_STACK_BUFFER_OVERRUN` (`0xC0000409`) after any cuDNN RNN forward: the
results are written correctly and *then* the interpreter crashes on the way
out, which would break `python run_seeds.py && python make_multiseed_table.py`
(the `&&` would never fire). Isolated to exactly this argument — the same
bidirectional GRU without it exits cleanly, and disabling cuDNN also avoids
it but would change the numerics and lose the fused-kernel speed.

The argument is therefore dropped in `run_seeds.py`, and only there.
**Verified inert before removing:** under the same seed, an `nn.GRU` built
with and without it has bitwise identical parameters *and* bitwise identical
outputs, in both train and eval mode, on both CPU and CUDA. Parameter counts
after the change are unchanged (764,290 / 734,274 / 787,074 / 751,874). The
real `nn.Dropout(DROPOUT_RATE)` layer is untouched.

### Verification performed (no dataset needed)

- All four architectures build, forward and backward on the RTX 5080; finite,
  non-zero gradients; correct output shape.
- `set_seed` reproducible: seed 42 twice → identical init; seed 42 vs seed 0 →
  different init; `cudnn.deterministic=True`, `benchmark=False` confirmed set.
- All 12 `CONFIGS` reference valid activations and architectures; no
  duplicates; 12 × 5 = 60 runs.
- `main()` exercised end-to-end on a tiny synthetic corpus in a temp dir:
  correct CSV schema, per-run append, running ETA, and a second invocation
  correctly resumed with **0 re-runs and no duplicate rows**. Exit code 0.

---

## Task 2 — Figures

*Addresses: Reviewer 3 point 1 (vector or ≥300 DPI, fully legible, no
clipping).*

### Added `make_figures.py` — **DONE**

Produces `assets/convergence_curves.pdf` and `assets/best_pairs.pdf` as
vector PDF via `savefig(format='pdf', bbox_inches='tight')`, plus
`assets/_preview_*.png` at 300 DPI for visual inspection (previews are not
referenced by the `.tex`).

- `font.family = serif` with Times New Roman first in the stack, matching the
  IEEEtran body text.
- `pdf.fonttype = 42` — fonts embed as subset TrueType. Verified: both PDFs
  contain `FontFile2` and `TimesNewRomanPSMT`, and **zero** Type 3 fonts.
- Sized for a single IEEE column and placed at the existing
  `width=0.92\columnwidth`, so **no `\includegraphics` width change is
  needed** — only the file extensions change from `.png` to `.pdf`.
- `save()` reads the trimmed `/MediaBox` back out of each written PDF,
  computes the real on-page scale factor, and reports the true rendered size
  of the smallest text, warning below 8 pt. Current output:
  Fig. 1 → **8.29 pt**, Fig. 2 → **8.60 pt**. Both clear the floor.
  (This check caught a real problem: Fig. 1's first draft had a 5-column
  legend that trimmed *wider* than the column, so LaTeX would have scaled it
  down to 7.68 pt. Fixed by tightening the legend handles and spacing.)
- **Fig. 1** — data from `results/convergence/convergence_per_af.csv`
  (`val_loss_e1..e10`). Ten homogeneous GRU pairs, distinct colour + marker
  per activation, legend placed *below* the axes so it cannot overlap the
  curves. Only the six epochs that actually ran are plotted.
- **Fig. 2** — data from
  `results/comparison/combination_vs_isolated_summary.csv` (`iso_max`,
  `combo_max`). Value labels and pair names sit fully inside the bars
  (white, rotated for the names); the legend sits in reserved headroom above
  the bars. y-axis truncated at 86.6 — **the caption must say "y-axis
  truncated"**.

### Caption bug — **PENDING** (needs the `.tex`, see Blockers)

The submitted Fig. 1 caption reads "Per-Epoch Validation *Accuracy*" but the
plot's y-axis is validation *loss*. Confirmed against the data: the
convergence CSV's `val_loss_*` columns are what the figure plots. The caption
must become "Per-Epoch Validation Loss (GRU, Homogeneous Pairs)", and the
sentence in Section IV-C that references the figure must say loss.

---

## Finding that changes Task 3d — seed inconsistency **resolved**

The Conclusion's *Seventh* finding says Mish (88.48 %) and SiLU (88.43 %)
were reached "under a different seed in the convergence rerun", while Table
VII's footnote says "Seed fixed at 42". Audited the scripts:

| Script | Seeding |
|---|---|
| `activation_sweep.py` (BiLSTM+LSTM) | **none** — only `train_test_split(random_state=42)` |
| `gru_activation_sweep.py` (BiGRU+GRU) | **none** — same |
| `bilstm_bigru_sweep.py` (both-bidirectional) | **none** — same |
| `convergence_rerun.py` | `torch.manual_seed(42)`, `np.random.seed(42)` |

So the claim is backwards. **Table VII's footnote is correct** — the
convergence rerun genuinely used seed 42. The 400-run sweep is the one that
was *unseeded*: its weight initialisation came from an uncontrolled RNG
state. The two runs differ because one is seeded and the other is not, not
because they used two different seeds.

The pipelines are otherwise byte-identical — `clean_text`, the split, the
vocabulary, `make_loaders`, the model, the optimiser, the scheduler and the
early-stopping rule all diff clean between `convergence_rerun.py` and
`gru_activation_sweep.py`. The only difference is the two seeding lines.

Measured effect on the ten homogeneous GRU pairs (unseeded sweep → seeded
rerun): **every one of the ten is higher**, by +0.14 to +0.78 points, mean
+0.48. A consistent one-sided shift of that size is what a single favourable
initialisation draw looks like when it is shared across all ten runs.

**Consequence for the Task 1 STOP gate:** the specified seed-42 sanity check
(“seed 42 must reproduce the original sweep number within ±0.05 %”) **cannot
pass**, because there is no seed-42 number to reproduce — the reference sweep
did not set a seed. `run_seeds.py` still performs and prints the comparison,
but reports it as a *delta to an unseeded reference* and states explicitly
that a mismatch does not imply a pipeline difference. Treat the printed mean
delta as the quantity of interest, not as a pass/fail.

---

## Experiment run — **DONE**, 60/60 runs

Dataset located at `D:\Research\dataset\IMDB Dataset.csv` (the original project
mirror) and copied to `dataset/`. Verified: 50,000 rows, 25k/25k balanced.
Run completed in ~13 min on the RTX 5080. Log: `results/run_seeds.log`.

### Headline: the single-seed rankings do not survive

Per-config accuracy std across 5 seeds is **0.15–0.48 (median 0.33)** — far
larger than the 0.07-point spread that separates the top-ten in Table V.

- **All 45 pairs of top-ten CIs overlap**, with a common overlap region
  (max lower bound 87.39 vs min upper bound 87.73). The top ten are
  **statistically indistinguishable from one another**.
- **Rank churn is severe**: mean absolute rank change 4.2 places, max 8.
  Table V's #1 (Tanh + Hardswish) falls to #9; Table V's #6 (ELU + PReLU)
  rises to #1.
- **Regression to the mean / winner's curse**: all **10 of 10** top configs
  score *lower* on rerun, by **−0.48 points on average** (−0.32 to −0.71).
  This is expected — the top ten were selected as maxima over 400 noisy
  single-draw runs, so their reported values are upward-biased.

### The ReLU + PReLU control (Section V-F) — magnitude does not reproduce

| | sweep (single draw) | 5-seed mean ± std | 95% CI |
|---|---|---|---|
| BiGRU+GRU | 87.35 | **87.58 ± 0.17** | 87.37–87.78 |
| BiLSTM+LSTM | 84.90 | **86.96 ± 0.30** | 86.59–87.33 |
| gap | **2.45** | **0.62** | CIs do **not** overlap |

The direction survives (CIs disjoint), but the magnitude collapses from 2.45
to 0.62 points. The sweep's 84.90 is **3.32 sd below its own sweep mean** and
sits **1.64 points below the lowest of five seeded reruns** (86.54).

**Pipeline independently verified** so this is not an artefact: rerunning
`ReLU + ReLU` on BiLSTM+LSTM (sweep cell 86.59) gives 86.93 ± 0.23, a delta of
+0.34 — normal. Saved to `results/pipeline_check.csv`. The BiLSTM+LSTM
pipeline is faithful; the sweep's 84.90 was an unlucky draw.

### Versus the 88.40% benchmark

No configuration's **mean** reaches it — best is ELU + PReLU at
87.70 ± 0.48 (95% CI upper bound 88.30). But **1 of 60 individual runs** did:
LeakyReLU + Tanh, BiGRU+GRU, seed 42, at **88.43**. So the benchmark is
reachable by a lucky single run but not by any configuration's expected
accuracy — which is a more precise version of the Conclusion's current claim.

### Seed-42 deltas vs the unseeded sweep

Mean +0.10, range −0.57 to +2.26; only 1 of 12 within ±0.05. As predicted,
the check cannot pass as originally specified because the reference sweep set
no seed. The +2.26 outlier is the ReLU + PReLU BiLSTM+LSTM cell above.

### Generated

- `results/multiseed_results.csv` — 60 rows
- `results/pipeline_check.csv` — 5 rows (verification)
- `tables/multiseed_table.tex` — 12 configs, sorted by mean accuracy

---

## RQ2 / Table VII seed-robustness check — **DONE**, survives

`check_convergence_seeds.py`, 10 homogeneous GRU pairs × 5 seeds = 50 runs.

**Pipeline fidelity note (a first attempt was wrong).** The initial check
reused `run_seeds.py`, which sets `cudnn.deterministic=True`. That flag —
correct for its own purpose — perturbs the numerics enough to flip epoch-1
outcomes, because epoch-1 accuracy is a knife-edge: the model either escapes
the ~50% plateau in one pass or it does not. Under that pipeline seed 42 gave
Mish 51.50 against Table VII's 81.08, and the finding appeared to collapse.
Caught by the seed-42 check. The corrected script replicates
`convergence_rerun.py` exactly (its `PaperGRU`, only `torch.manual_seed` +
`np.random.seed`, no cudnn flags) and **reproduces Table VII bit-for-bit at
seed 42 — max deviation 0.00 points across all ten activations.**

Results on the correct pipeline:

| | value |
|---|---|
| median per-AF seed std (epoch-1 acc) | **4.39** points |
| spread across the ten activations | **15.30** points = **3.5×** the noise |
| Mish vs ReLU gap | **+12.52** (single-seed claimed +17.23) = **2.9×** noise |
| smooth family vs ReLU family | 73.70 vs 61.05 = **12.65** points |
| Spearman rho vs Table VII ordering | **0.891** (p = 0.0005) |
| runs still at the ~50% plateau | 2 / 50 |

**Verdict: RQ2 holds.** The ordering is strongly preserved and the
smooth-vs-ReLU gap is ~3× seed noise. One honest caveat: seed 42 is the
favourable end — Table VII's printed value is the maximum of the five seeds
for 6 of 10 activations, so the *magnitude* (17.23) is optimistic while the
*conclusion* is sound. Recommend quoting the 5-seed means alongside.

Output: `results/convergence_seeds.csv`

---

## Full 5-seed sweep, Batch 1: BiLSTM+LSTM — **DONE**, 500/500

`run_full_sweep.py --arch "BiLSTM+LSTM" --seeds 10 20 30 40 50`
100 cells × 5 seeds = 500 runs, 107 min, 5 parallel workers (one per seed),
GPU held at 89% utilisation. All workers exit 0.

Output: `BiLSTM+LSTM_5_seed_variance_results/` — `seed_{10..50}.csv` (100 rows
each), `all_runs.csv` (500), `mean_100_cells.csv` (100, mean ± std ± 95% CI),
`diff_vs_single_seed.csv`, `run.log`.

**Parallelism verified non-perturbing**: two cells re-run serially afterwards
reproduce the parallel results exactly (delta 0.0000 on both).

### What changed vs the single-seed sweep

| level | single seed | 5-seed | verdict |
|---|---|---|---|
| architecture mean | 86.70 | **86.62** | **stable (−0.08)** |
| per-cell ranking | — | — | **no relationship** |

- Mean absolute cell movement **0.53 points**; 11/100 cells moved >1.0.
- **Top-10 overlap: 0 of 10.** Bottom-10 overlap: 1 of 10.
- Spearman rho over 100 cells = **−0.174**; mean rank change **37 places**,
  max 97.
- Paper's BiLSTM+LSTM champion Mish + PReLU (87.66) is actually **rank
  87/100** at 86.14 ± 0.?? — the single seed over-rated it by 1.52.

### Claims that do not survive (BiLSTM+LSTM only)

**RQ6 / Table X — collapses.** ReLU + PReLU: 84.90 (rank 99/100, "second
lowest of all 400") → **86.76 ± 0.42, rank 35/100**. Gap below the
architecture peak: claimed 2.76 → **0.40**. **Zero of 500 runs fell below
60% accuracy**, so there is no dead-neuron failure to explain. Worse for the
narrative: **ReLU is now the best Act1 in this architecture** (86.83, rank
1/10; the paper had it 9th at 86.50).

**RQ5, LSTM half — direction reverses, significantly.** Paper: "Tanh 86.95
beats ReLU 86.50 by 0.45". 5-seed: **ReLU 86.83 > Tanh 86.61**, t = 2.07,
p = 0.041.

**RQ4, LSTM preferences — wrong.** "LSTM works best with Tanh and Mish as
Act1" → Tanh is **7th**, Mish is **9th** of 10.

**Conclusion, third finding.** "ELU is … the worst for LSTM" → ELU is now
**3rd best** Act1 (86.74).

Table VI BiLSTM+LSTM column rank correlation with the paper: Act1
rho = **−0.445**, Act2 rho = +0.200.

### What survives, and one new finding

- Architecture-level mean is stable (−0.08), so **Table IV and the GRU > LSTM
  conclusion are unaffected**.
- The BiLSTM+LSTM grid is **flat**: excluding one unstable cell, the 100 cell
  means span 86.17–87.16 (~1 point) against a median per-cell std of 0.53.
  Activation choice barely matters in this architecture; the single seed
  manufactured structure that is not there.
- **New, genuine finding**: `Hardswish + LeakyReLU` is truly unstable —
  mean 82.06, **std 9.27**, range 65.5–86.85. The only such cell in 100.

---

## Full 5-seed sweep, Batch 2: BiGRU+GRU — **DONE**, 500/500

78 min, 5 workers, GPU 89%. Output: `BiGRU+GRU_5_seed_variance_results/`.

| level | single seed | 5-seed | verdict |
|---|---|---|---|
| architecture mean | 87.43 | **87.48** | **stable (+0.04)** |
| per-cell ranking | — | — | **no relationship** |

- Per-cell std median **0.33**; **no training failures** (0/500 below 60%,
  no cell with std > 2) — markedly more stable than BiLSTM+LSTM.
- The 100 cell means span only **0.69 points** (87.12–87.81).
- **Top-10 overlap: 1 of 10.** Spearman rho over 100 cells = **−0.059**;
  mean rank change **34.4** places.
- Table VI Act1 marginal rho = **−0.383**, Act2 = +0.182.
- Table IV min for this row was 85.68 single-seed → **87.12** over 5 seeds:
  the low tail was noise, not a real failure mode.

**RQ4/RQ5 GRU half breaks.** "GRU works best with ELU (87.69) and LeakyReLU
(87.60) as Act1" → **ELU is now 9th of 10** (87.44); the best Act1 is PReLU
(87.55), which the paper ranked 8th. ELU appears in **zero** of the new top-10.

**RQ6 control.** ReLU + PReLU: 87.35 → 87.31 ± 0.21 (rank 86/100) — this one
barely moved, unlike its BiLSTM+LSTM twin.

---

## Combined verdict after 1000 runs (both architectures)

### Confirmed, and stronger than published

**GRU > LSTM.** 87.48 ± 0.03 vs 86.62 ± 0.10 (95% CI, n = 500 each).
Gap **+0.85** points, Welch t = 16.3, **p = 1.5e-49**. The paper claimed +0.73
from a single seed; the true gap is larger and now essentially certain.

**Architecture dominates activation.** The GRU-vs-LSTM gap (0.85) is larger
than the *entire* spread of all 100 activation pairs within BiGRU+GRU (0.69).
This is a cleaner statement of the paper's real contribution than anything
currently in it.

### Refuted

**"The best activation for one architecture is often the worst for another"**
(Abstract, RQ4, Conclusion finding 3). Rank correlation of the 5-seed
marginals between the two architectures:

| position | rho | p | meaning |
|---|---|---|---|
| Act1 | **−0.164** | 0.651 | uncorrelated, **not** inverted |
| Act2 | **+0.806** | **0.005** | significantly **positively** correlated |

Act2 rankings actually *agree* across architectures — the opposite of the
claim. The paper's own example ("ELU is the best Act1 for GRU but the worst
for LSTM") reverses: ELU is **9th in GRU and 3rd in LSTM**.

### Status of every headline claim

| claim | verdict |
|---|---|
| GRU > LSTM | **confirmed, p = 1.5e-49** |
| All ten best configs use GRU | holds (GRU mean is higher everywhere) |
| Architecture > activation | **new, strongly supported** |
| Convergence: smooth AFs start faster | holds (rho 0.891, 3.5× noise) |
| Top-10 ranking / "best pair" | **refuted** (rho ≈ 0 both architectures) |
| Rankings invert across architectures | **refuted** (Act2 positively correlated) |
| RQ6 dead neurons | **refuted** (0/1000 failures; ReLU is LSTM's best Act1) |
| RQ5 non-saturating vs Tanh | **refuted** in LSTM (direction reversed, p = 0.041) |

---

## Full 5-seed sweep, Batches 3 & 4 — **DONE**. All 2000 runs complete.

`BiLSTM+BiLSTM` (105 min) and `BiGRU+BiGRU` (81 min), chained so the GPU never
idled. 4 architectures × 100 cells × 5 seeds (10/20/30/40/50) = **2000 runs**.

### NEW TABLE IV — every architecture mean is unchanged

| Experiment | Min% | Max% | **Mean%** | 95% CI | old Mean | diff |
|---|---|---|---|---|---|---|
| BiLSTM+LSTM | 82.06 | 87.16 | **86.62** | ±0.10 | 86.70 | −0.08 |
| BiGRU+GRU | 87.12 | 87.81 | **87.48** | ±0.03 | 87.43 | +0.04 |
| BiLSTM+BiLSTM | 86.07 | 87.14 | **86.75** | ±0.05 | 86.79 | −0.04 |
| BiGRU+BiGRU | 87.14 | 87.73 | **87.48** | ±0.03 | 87.30 | +0.18 |

### Per-architecture reliability — the same verdict four times

| Experiment | median std | cell spread | top-10 overlap | Spearman rho | failures |
|---|---|---|---|---|---|
| BiLSTM+LSTM | 0.53 | 5.11 | 0/10 | −0.174 | 0/500 |
| BiGRU+GRU | 0.33 | 0.69 | 1/10 | −0.059 | 0/500 |
| BiLSTM+BiLSTM | 0.48 | 1.07 | 1/10 | −0.041 | 0/500 |
| BiGRU+BiGRU | 0.29 | 0.58 | 0/10 | +0.055 | 0/500 |

Four independent architectures, four independent rho values, all ≈ 0. The
single-seed cell ranking carried no information in any of them. **Zero
training failures in 2000 runs.**

### Confirmed and strengthened

**GRU > LSTM** (1000 runs per family): 87.48 ± 0.02 vs 86.69 ± 0.05, gap
**+0.79**, Welch t = 26.7, **p = 2.3e-125**.

**Every one of the global top 50 cells is GRU-based (50/50).** The paper
claimed this for the top 10; it holds five times over.

**Bidirectionality in layer 2 barely matters** (paper: "changes little"):
BiLSTM+BiLSTM − BiLSTM+LSTM = **+0.13** (p = 0.017, small but real);
BiGRU+BiGRU − BiGRU+GRU = **+0.00** (p = 0.858). The paper's −0.13 for the
GRU pair was noise; the true difference is zero.

### The benchmark, settled

**0 of 400 cells** have a mean above 88.40. **0 of 2000 individual runs**
reached 88.40 — the maximum single run was **88.39**. Combined with the
earlier 60-run experiment (1 run at 88.43 under seed 42), 88.40 is reachable
roughly once in 2000 runs. The Conclusion's "Mish and SiLU reached 88.48 and
88.43" describes two such outliers, not typical behaviour.

### New global top-10 (replaces Table V)

All ten GRU-based, spanning **0.12 points**, widest CI ±0.50 — with a common
overlap region, so **all ten are statistically indistinguishable**.

| # | Act1 + Act2 | Architecture | Acc mean ± std |
|---|---|---|---|
| 1 | ReLU + LeakyReLU | BiGRU+GRU | 87.81 ± 0.17 |
| 2 | SiLU + Mish | BiGRU+BiGRU | 87.73 ± 0.16 |
| 3 | Tanh + LeakyReLU | BiGRU+BiGRU | 87.72 ± 0.40 |
| 4 | Tanh + GELU | BiGRU+GRU | 87.71 ± 0.29 |
| 5 | PReLU + ReLU | BiGRU+GRU | 87.71 ± 0.17 |

Not one of the paper's four per-architecture champions survives: they now rank
98/100, 77/100, 91/100 and 85/100 in their own grids.

### Rankings neither transfer nor invert

All six pairwise Spearman correlations of the Act1 marginals are
non-significant (rho −0.164 to +0.576, p = 0.082 to 0.751). The Abstract's
"best for one architecture is often the worst for another" is not supported;
neither is its opposite. Activation rank is simply noise-dominated.

Output: `results/all_400_cells_5seed.csv` (all 400 cells, mean ± std ± CI).

---

## Defect found in the supplied `.tex`

Table IV (`tab:sweep_summary`) rows 1–2 read `BiLSTM+BiLSTM+LSTM` and
`BiGRU+BiGRU+GRU`. The submitted PDF reads `BiLSTM+LSTM (Exp.~1)` and
`BiGRU+GRU (Exp.~2)`. A find/replace corrupted the reconstruction. Will be
fixed in the camera-ready to match the PDF. No other doubled labels found.

---

## Tasks 3–7 — **PENDING**

Source `.tex` now available. Task 3 needs a decision from the author on how
far to qualify RQ1 and RQ6 (see report). Tasks 4/5/7 additionally need a
LaTeX distribution — `pdflatex` is still not installed (`winget` and `choco`
are both available).

---

## Blockers — files that do not exist anywhere on this machine

Searched the whole user profile. None of the following are present, and none
are in git (`Paper/` and `dataset/` are both in `.gitignore`, so they were
never committed):

| Missing | Blocks |
|---|---|
| `Iconeect2026_reconstructed-1.tex` | Tasks 3, 4, 5, 6, 7 — every text edit |
| `Iconeect2026_paper_final.pdf` | Reference for what the reviewers saw |
| `assets/convergence_curves.png`, `assets/best_pairs.png` | Nothing — replaced by the new PDFs |
| `dataset/IMDB Dataset.csv` | **Running** `run_seeds.py` |
| a LaTeX distribution (`pdflatex` not on `PATH`) | Tasks 4, 5, 7 — compiling and page-count checks |

The coding work did not need the `.tex`: the top-10 configurations were given
explicitly in the task brief and were independently verified against the sweep
CSVs, and the Table III hyperparameters were taken from the sweep scripts,
which are the actual source of those values.

---

# CAMERA-READY BUILD (9 September 2026)

## Conference requirements, fetched from the official site

Source: `iconeect2026.puc.ac.bd` camera-ready guidelines page.

| Requirement | Status |
|---|---|
| 4-6 pages, cannot exceed 6 | **6 pages** |
| A4 or US Letter | **A4** (595x842 pt) |
| File size < 5 MB | **0.32 MB** |
| All fonts embedded, no Type 3 | **21 fonts, all embedded, 0 Type 3** |
| First-page header, 9 pt, top left | present, page 1 only |
| First-page copyright footer | present, page 1 only |
| IEEE conference template | `\documentclass[conference,a4paper]{IEEEtran}` |
| Similarity < 30% | **author must run this** |
| PDF eXpress, conference ID `71331X` | **author must run this** |
| Response letter (mandatory) | `PID_Response.pdf`, 3 pages |

Submission portal: `cmt3.research.microsoft.com/iCONEECT2026`.
Required file names and the full checklist are in
`camera_ready_submission/README.md`.

## Deliverables

- `Iconeect2026_camera_ready.tex` / `.pdf` (original `.tex` untouched)
- `PID_Response.tex` / `.pdf`
- `camera_ready_submission/` with `PID.pdf`, `PID_Response.pdf`, `source/`
- `analysis/FINAL_mean_table_2000runs.csv` (400 cells)
- `analysis/FINAL_variance_vs_single_seed.csv` (400 cells, per-cell deltas)
- `analysis/FINAL_results_summary.md`
- `assets/convergence_curves.pdf`, `assets/architecture_spread.pdf`
- `verify_camera_ready.py`, `make_paper_tables.py`, `make_figures.py`

## Paper changes, top to bottom

**Title** now leads with the contribution: *How Much Does the Activation
Function Matter? A 2,000-Run Variance Study of Gated Recurrent Networks for
Sentiment Classification*.

**Author block** de-anonymised with placeholders for three authors
(`[AUTHOR n NAME]`, `[DEPT n]`, `[INSTITUTION n]`, `[CITY n]`, `[COUNTRY n]`,
`[EMAIL n]`). Note: a `\` immediately followed by `[CITY 1]` is parsed by
LaTeX as `\[length]`; the placeholders are therefore written `\{}` and the
same care is needed when substituting real values.

**Abstract** rewritten to lead with the 2,000-run scale and the variance
split, then the surviving architectural result.

**Introduction** reframed around replication rather than a leaderboard, with
the six RQs restated as questions about measurability. States the exclusivity
claim directly: largest controlled activation study in gated recurrent
networks and the first in this setting to report seed variance.

**Literature Review** gains a fourth gap, now the primary one: none of
[eger2018, farzad2019, essaiAli2022, zhu2022, ahmed2025] repeats any
configuration or reports a standard deviation, a confidence interval or a
seed. Paragraphs 2-3 compressed ~25% with every citation retained.

**Methodology** gains Section III-E, Replication Protocol. Table III now
lists the fixed split seed (42) and the five training seeds separately.
Section III-F rewritten: the 88.40% benchmark sits outside the 95% interval
of our best configuration and was not reached by any of 2,000 runs.

**Results** rebuilt. New Table IV (variance decomposition) leads. Table V is
the top-ten with sigma and 95% CI (Reviewer 3, point 2). Table VI is the
single-seed vs 5-seed agreement. Table VII is the 5-seed convergence study.
New subsections on reproducibility and on reliability/failure modes.

**Discussion** condensed and re-scoped to the new RQs, ending with the power
analysis (4 seeds for the architecture effect, ~157 for the top-ten ordering).

**Conclusion** rewritten to five findings; future work names SST-2 and Amazon
Reviews specifically, answering Reviewer 3 point 3.

## Tables and figures

Removed: former Table VIII (precision-recall balance) and Table X (ReLU
dead-neuron) - both refuted by replication. Former Table IV (architecture
summary) removed as redundant with the new Fig. 1. Marginal-means table
replaced by prose plus the key statistics.

**Fig. 1** is new: all 400 replicated cell means grouped by architecture with
95% CIs. Replaces the former Fig. 2 bar chart, whose single-seed peaks do not
reproduce. **Fig. 2** is the convergence plot, now mean over five seeds with
+/-1 sd bands, and its caption corrected from "Accuracy" to "Loss".

Both are vector PDF with TrueType subsets embedded. `make_figures.py` reads
the trimmed MediaBox back out of each written PDF and verifies on-page text
size: Fig. 1 8.60 pt, Fig. 2 8.28 pt, both above the 8 pt floor.

## Page budget

Started at 7 pages, reached 6 by, in order: removing the marginal-means table;
trimming the Introduction, Replication Protocol, Section III-F, Discussion and
Conclusion; removing the redundant architecture-summary table; reducing figure
heights (width unchanged, so font size unaffected); and setting the
bibliography in `ootnotesize` with `\itemsep -1pt`. The reference list
itself is unchanged.

Note: an earlier `sed` intended to apply `ootnotesize` to the bibliography
silently did nothing, because `{23}` in `egin{thebibliography}{23}` is a BRE
interval expression. Applied correctly afterwards.

## 60-run experiment removed

Per instruction, the 12-config x 5-seed pilot is deleted:
`results/multiseed_results.csv`, `results/run_seeds.log`,
`results/pipeline_check.csv`, `tables/multiseed_table.tex`,
`make_multiseed_table.py`, and `run_seeds.py`. The shared training pipeline
that `run_full_sweep.py` imports was extracted from `run_seeds.py` first and
now lives in `pipeline.py` with the pilot-specific configuration stripped out.

## Verification

`verify_camera_ready.py` checks page count, page size, file size, font
embedding, Type 3 fonts, unresolved references, placeholders, header/footer
placement and page numbers. All blocking checks pass; the only warning is the
18 author placeholders, which is expected until the author block is filled in.

Two defects were found in the verifier itself and fixed: scanning raw PDF
streams for `??` produced false positives from embedded font binaries (now
uses rendered text), and `ancyhf{}` in the first-page style was clearing
IEEEtran's `\IEEEpubid` footer (the notice is now printed via `ancyfoot`
while `\IEEEpubid` still reserves the column space).

All 31 numeric values quoted in the paper's prose were cross-checked against
`tables/numbers.txt`, which is generated from the CSVs. Every table is
generated by `make_paper_tables.py`; no number is typed by hand.

## Still on the author

1. Fill in the author block, delete unused blocks, recompile, re-check pages.
2. Run a similarity check (<30%).
3. IEEE PDF eXpress with conference ID `71331X`; save output as `PID.pdf` and
   the confirmation as `PID_pdfeXpress.pdf`.
4. Sign the IEEE eCF in CMT, save as `PID_copyright.pdf`.
5. Register and pay; save receipt as `PID_payment.pdf`.

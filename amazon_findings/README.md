# Amazon Polarity Cross-Dataset Experiment — Findings

Testing whether the ten activation pairs that lead the 2,000-run IMDb sweep
still lead on a second sentiment corpus, under the IMDb pipeline's own
settings.

**Status: complete.** 10 pairs × 4 architectures × 5 seeds = **200/200 runs**.

---

## TL;DR

1. **The headline result is not about activations.** It is that the paper's
   architecture silently fails on short-review corpora. Two of the four
   architectures collapse to chance on Amazon Polarity; the other two are
   completely unaffected. The split is exactly whether the **second recurrent
   layer is bidirectional**.

2. **The IMDb top-10 ordering does not transfer.** Spearman ρ = **−0.079**
   (p = 0.83) — no relationship at all. But this number should not be read as
   "activations don't generalise", because 74 of the 200 runs never learned
   anything, so most pair means encode *how many seeds survived*, not
   activation behaviour.

3. **Where training worked, the ten pairs are indistinguishable.** In the two
   unaffected architectures every run trained cleanly, and the ten pairs span
   **0.32** and **0.28** accuracy points respectively — smaller than their own
   seed-to-seed SDs.

---

## The experiment

| | |
|---|---|
| Corpus | `mteb/amazon_polarity` @ `e2d317d38cd51312af73b3d32a06d1a08b442046` |
| Full size | 3,600,000 train / 400,000 test |
| Subsample | 50,000 stratified, `random_state=42` → 40,000 train / 10,000 test |
| Preprocessing | `clean_text`, 5,000-token train vocab, PAD=0, UNK=1, MAX_LEN=200 |
| Padding | **right / post** — first 200 tokens kept, zeros appended (matches `pipeline.py`) |
| Pairs | the 10 leading `results/master_2000_runs.csv`, imported from `amazon_polarity_sweep.TOP10` |
| Architectures | BiLSTM+LSTM, BiGRU+GRU, BiLSTM+BiLSTM, BiGRU+BiGRU |
| Seeds | 10, 20, 30, 40, 50 |
| Training | imported wholesale from `pipeline.py` — Adam 1e-3, `ReduceLROnPlateau`, dropout 0.4, grad-clip 1.0, ≤10 epochs, early-stop patience 3 |

The subsample to 50,000 is deliberate: it matches IMDb's exact 40k/10k shape,
so the corpus is the only variable that moves. Training on all 3.6M rows would
have confounded corpus identity with a 90× change in data volume.

No model change of any kind — no packing, no masking, no pooling change.
Everything except the data loading is imported from `pipeline.py`.

---

## Finding 1 — the architecture split

| architecture | 2nd layer | readout | escaped | at chance | mean acc | SD |
|---|---|---|---|---|---|---|
| BiLSTM+LSTM | unidirectional | `h[-1]` | 5/50 | 41/50 | 54.79 | 11.40 |
| BiGRU+GRU | unidirectional | `h[-1]` | 16/50 | 33/50 | 63.29 | 18.78 |
| BiLSTM+BiLSTM | **bidirectional** | `cat(h[0], h[1])` | **50/50** | 0/50 | **90.14** | 0.43 |
| BiGRU+BiGRU | **bidirectional** | `cat(h[0], h[1])` | **50/50** | 0/50 | **90.60** | 0.24 |

"escaped" = reached ≥80%; "at chance" = ≤55%. Across all 200 runs: 121 learned,
74 sat at chance, and **30 landed at exactly 50.00%** — predicting a single
class for every input, with precision or recall of 0.0.

### Why

Every model right-pads to 200 tokens and classifies from a final hidden state.
IMDb reviews are long (median **176** of 200 slots filled), so the readout sits
~24 steps past the text. Amazon Polarity reviews are short (median **71**),
leaving ~**129** steps of zero-embedding input before the readout.

A padding step is not a no-op. With `x_t = 0` a GRU still fires its gates:

```
z_t = σ(W_hz·h + b_z),  r_t = σ(W_hr·h + b_r)
n_t = tanh(b_in + r_t ⊙ (W_hn·h + b_hn))
h_t = (1 − z_t)⊙n_t + z_t⊙h_{t−1}
```

That is a fixed autonomous map `h → F(h)`, and at initialisation
(`U(−1/√64, +1/√64)`) it is strongly contractive. Applied ~129 times it drives
any two reviews to the same fixed point. The classifier then reads a vector
that is effectively constant across the dataset, and the best available
strategy is the base rate — 50%, at loss `ln 2 = 0.6931`. The same 129
Jacobians vanish the gradient on the way back, so the model cannot learn its
way out either.

**Why the bidirectional variants are immune:** `FullBiLSTM`/`FullBiGRU` read
`torch.cat([h[0], h[1]])`, where `h[1]` is the **backward** direction's final
state. The backward RNN runs right-to-left — it consumes the trailing padding
*first*, while its state is still at initialisation, then reads the text, and
finishes on token 0. Its readout is adjacent to real content no matter how much
padding trails the sequence. The forward half still dies; the backward half
carries the signal, and that is enough.

So the flaw is specific to architectures whose **second** recurrent layer is
unidirectional. Half the grid was never at risk.

### Controlled check

`diagnose_amazon.py` — BiGRU+GRU / ReLU+ReLU / seed 10, 3 epochs, only the
padding moved (`logs/padding_diagnostic.log`):

| condition | val. accuracy |
|---|---|
| Amazon, right-pad 200 (readout ~129 steps late) | **49.83%**, loss 0.6931 |
| Amazon, left-pad 200 (readout 0 steps late) | **90.75%** |
| Amazon, right-pad 100 (readout ~29 steps late) | **90.20%** |
| IMDb, right-pad 200 (control) | 86.72% |

The third row is the decisive one: still right-padded, just a shorter tail. It
is the **distance from text to readout** that matters, not padding as such.

---

## Finding 2 — the ranking does not transfer

Overall, accuracy averaged with equal weight across the 4 architectures and 5
seeds (20 runs per pair):

| rank | pair | Amazon | SD | escaped | IMDb | IMDb rank | Δrank |
|---|---|---|---|---|---|---|---|
| 1 | ReLU+Tanh | 78.73 | 18.06 | 14/20 | 87.29 | 3 | +2 |
| 2 | GELU+ELU | 78.19 | 18.68 | 14/20 | 87.26 | 5 | +3 |
| 3 | ReLU+Hardswish | 76.65 | 18.46 | 13/20 | 87.23 | 10 | +7 |
| 4 | PReLU+SiLU | 76.31 | 19.54 | 13/20 | 87.24 | 9 | +5 |
| 5 | LeakyReLU+Mish | 76.16 | 19.69 | 13/20 | 87.25 | 6 | +1 |
| 6 | ReLU+GELU | 74.19 | 20.26 | 12/20 | 87.31 | 1 | −5 |
| 7 | PReLU+ReLU | 72.64 | 20.01 | 11/20 | 87.25 | 7 | 0 |
| 8 | ReLU+ReLU | 72.14 | 20.48 | 11/20 | 87.29 | 2 | −6 |
| 9 | LeakyReLU+GELU | 71.61 | 20.13 | 10/20 | 87.26 | 4 | −5 |
| 10 | LeakyReLU+ReLU | 70.42 | 20.62 | 10/20 | 87.24 | 8 | −2 |

**Spearman ρ = −0.079, p = 0.83.** Mean |rank shift| 3.6 places. Accuracy level
IMDb 87.26% → Amazon 74.70% (−12.56).

### Read this before using that ρ

The `escaped` and `Amazon` columns are the same column. Rank 1 has 14 surviving
runs, rank 10 has 10. **The overall ordering is a re-encoding of how many of
each pair's 20 runs happened to escape the plateau**, which is a property of
seed luck under a broken readout, not of the activation pair. Every SD is
≈ 18–21 points, because each mean averages ~50% failures with ~90% successes.

The honest statement is: *under right-padding, the IMDb top-10 ordering does
not reproduce on Amazon Polarity, but the experiment cannot attribute that to
activation behaviour, because the dominant effect is an architecture/corpus
interaction that destroys training in half the grid.*

### Where training actually worked

In the two unaffected architectures every one of the 50 runs trained properly,
so these rankings are the only ones reflecting activation behaviour:

**BiLSTM+BiLSTM** — span **0.32** points, seed SDs 0.23–0.72

| rank | pair | accuracy |
|---|---|---|
| 1 | PReLU+SiLU | 90.30 ± 0.41 |
| 2 | LeakyReLU+ReLU | 90.27 ± 0.23 |
| 3 | ReLU+GELU | 90.23 ± 0.32 |
| 4 | GELU+ELU | 90.22 ± 0.42 |
| 5 | LeakyReLU+Mish | 90.21 ± 0.50 |
| 6 | ReLU+Hardswish | 90.10 ± 0.39 |
| 7 | PReLU+ReLU | 90.07 ± 0.48 |
| 8 | ReLU+Tanh | 90.06 ± 0.51 |
| 9 | LeakyReLU+GELU | 90.00 ± 0.72 |
| 10 | ReLU+ReLU | 89.98 ± 0.47 |

**BiGRU+BiGRU** — span **0.28** points, seed SDs 0.13–0.37

| rank | pair | accuracy |
|---|---|---|
| 1 | LeakyReLU+ReLU | 90.74 ± 0.37 |
| 2 | LeakyReLU+GELU | 90.70 ± 0.25 |
| 3 | ReLU+ReLU | 90.69 ± 0.29 |
| 4 | PReLU+SiLU | 90.69 ± 0.27 |
| 5 | ReLU+Tanh | 90.58 ± 0.15 |
| 6 | PReLU+ReLU | 90.56 ± 0.30 |
| 7 | ReLU+Hardswish | 90.54 ± 0.18 |
| 8 | GELU+ELU | 90.54 ± 0.25 |
| 9 | ReLU+GELU | 90.53 ± 0.13 |
| 10 | LeakyReLU+Mish | 90.46 ± 0.20 |

Both spans are **smaller than the seed SD of a single entry**. The ten pairs are
not separable on this corpus. Note also that the two rankings barely agree with
each other — ReLU+ReLU is last in one and third in the other.

This mirrors IMDb, where the same ten span 0.070 points against a median
standard error of 0.102.

---

## Finding 3 — no Amazon corpus matches IMDb's length profile

| corpus | median tokens | % filling 200 slots |
|---|---|---|
| **IMDb (target)** | **176** | **42.1%** |
| Amazon Polarity | 71 | 0.1% |
| Amazon-2023 Books | 70 | 24.5% |
| Amazon-2023 CDs_and_Vinyl | 68 | 27.0% |
| Amazon-2023 Kindle_Store | 58 | 11.0% |
| Amazon-2023 Video_Games | 44 | 11.9% |
| Amazon-2023 Movies_and_TV | 27 | 7.9% |

Not one comes close. IMDb is long-form film criticism; Amazon reviews are a few
sentences about a product. The mismatch is intrinsic to the domain, so swapping
Amazon categories will not avoid the problem — only length-filtering, changing
`MAX_LEN`, or fixing the readout would.

---

## Files

```
amazon_findings/
├── README.md                           this document
├── build_findings.py                   regenerates everything below
├── data/
│   ├── all_runs.csv                    200 runs, one row each (primary data)
│   ├── summary_by_pair_arch.csv        mean/SD of acc, prec, rec, F1 per (arch, pair)
│   ├── summary_overall.csv             overall accuracy per pair
│   ├── ranking_within_architecture.csv 40 rows: rank of each pair within each arch
│   ├── ranking_overall_vs_imdb.csv     overall ranking + IMDb comparison + Δrank
│   ├── imdb_vs_amazon_top10.csv        as emitted by the experiment script
│   ├── escape_rate_by_architecture.csv the Finding 1 table
│   ├── corpus_length_profile.csv       the Finding 3 table
│   ├── headline_numbers.json           the numbers quoted in this README
│   └── run_config.json                 full config + cache fingerprint
├── runs/
│   ├── rightpad_seed_{10..50}.csv      per-seed raw output (40 runs each)
│   └── leftpad_partial_seed_{10..50}.csv   see caveats
└── logs/
    └── padding_diagnostic.log          the controlled padding check
```

Column meanings in `all_runs.csv`: `act1`, `act2` (dense-layer activations),
`experiment` (architecture), `seed`, `accuracy`, `precision`, `recall`, `f1`
(test-set, %), `best_epoch` (epoch with lowest val loss, weights restored from
it), `stop_epoch` (where early stopping fired), `cell` (`act1+act2`).

`best_epoch` is diagnostic: collapsed runs peak at epoch 1–2 and stop at 4,
while runs that escaped use most of the 10 epochs.

---

## Caveats

- **The left-padded run is incomplete — 30 of 200 runs**, BiLSTM+LSTM only
  (`runs/leftpad_partial_*.csv`). It was stopped early and is included only for
  reference. Do not quote statistics from it.
- **Only the top 10 pairs were run.** There is no contrast set, so this cannot
  show the ten are still *better than the other 90* on Amazon — only how they
  re-order among themselves. All rankings here are explicitly *within the
  selected 10*.
- **The corpus was subsampled to 50,000** to match IMDb's shape. Results at full
  3.6M scale are unknown.
- **The IMDb comparison column** comes from `results/master_2000_runs.csv`,
  which was produced under `cudnn.deterministic=True`; this experiment inherits
  the same setting via `pipeline.set_seed`.
- The near-chance results are **reported as measured**. Padding was not switched
  and no setting was tuned to improve them.

---

## Reproducing

```bash
python download_amazon.py                    # fetch corpus via MTEB (~1 GB)
python amazon_test_top_10.py --prepare       # build + verify right-padded cache
python amazon_test_top_10.py                 # 200 runs, 5 GPU workers
python amazon_test_top_10.py --analyze       # tables
python amazon_findings/build_findings.py     # regenerate this folder's CSVs
```

`cache/` is git-ignored and regenerable. The sweep is resumable and skips
completed `(pair, architecture, seed)` rows; it refuses to resume against a
changed config or a different cache fingerprint.

Verification run before training confirmed: correct shapes, token IDs in
`[0, 5000)`, padding a contiguous **suffix** with zero rows beginning in
padding, no all-padding rows, balanced labels, and — against the left-padded
cache — identical labels and identical token sequences in identical order while
the arrays themselves differ. Same data, re-padded, nothing reused.

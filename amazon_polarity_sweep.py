"""
=============================================================================
CROSS-DATASET PERSISTENCE OF THE TOP-10 ACTIVATION PAIRS
Amazon Polarity (MTEB AmazonPolarityClassification)
=============================================================================
The 2,000-run sweep ranked all 100 activation pairs on IMDb. This asks
whether the pairs at the top of that ranking are still at the top on a
different sentiment corpus, under an otherwise identical protocol.

    10 pairs  x  4 architectures  x  5 seeds  =  200 runs

-----------------------------------------------------------------------------
WHAT IS HELD IDENTICAL
-----------------------------------------------------------------------------
Everything except the corpus. The architectures, activation registry,
set_seed(), the train/eval loop, the optimiser, the scheduler, grad clipping,
the early-stopping rule and the metric computation are IMPORTED from
pipeline.py rather than restated, so they cannot drift from the IMDb sweep.

Preprocessing is reproduced step for step from rs.load_data():
  clean_text (strip HTML, drop non-alphabetic, lowercase)
  -> 80/20 train/test split, stratified, random_state=42
  -> vocabulary of the 5,000 most frequent TRAIN tokens, <PAD>=0, <UNK>=1
  -> truncate/zero-pad to 200 tokens
The split is fixed at 42 and is never a function of the seed, exactly as in
the IMDb sweep, so the reported SDs isolate initialisation variance.

-----------------------------------------------------------------------------
DELIBERATE DEVIATION 1: PADDING SIDE  (required -- see encode())
-----------------------------------------------------------------------------
Sequences are LEFT-padded here, where the IMDb pipeline post-pads. This is
not a tuning choice: with post-padding every run on this corpus collapses to
49.8% with loss frozen at ln 2, because the models read the final hidden
state and Amazon's short reviews leave ~129 steps of zero input before it.
No Table III hyperparameter changes. Full measurement in diagnose_amazon.py.

-----------------------------------------------------------------------------
DELIBERATE DEVIATION 2: CORPUS SIZE
-----------------------------------------------------------------------------
Amazon Polarity ships 3.6M train / 400k test rows -- 90x the IMDb corpus.
Training 200 models on it at that size is weeks of GPU time, and it would
confound the comparison anyway: any change in the ranking could then be a
consequence of 90x more data rather than of the corpus itself.

We therefore draw a stratified subsample matched to IMDb's exact shape:
50,000 reviews -> 40,000 train / 10,000 test, balanced, random_state=42.
"Same environment" is read as same data scale as well as same settings, so
that the corpus is the only variable that moves.

-----------------------------------------------------------------------------
PARALLELISM
-----------------------------------------------------------------------------
One worker process per seed, five concurrent, as in run_full_sweep.py. Each
worker owns its own CUDA context, its own RNG and its own output file, and
calls rs.set_seed() immediately before building each model. With
cudnn.deterministic=True and benchmark=False, a given (pair, arch, seed) is
identical no matter which worker runs it or what runs alongside it.

Resumable: each worker skips rows already in its CSV and fsyncs after every
run, so an interrupt costs at most one run.

Usage:
    python amazon_polarity_sweep.py --prepare      # build the encoded cache
    python amazon_polarity_sweep.py                # run all 5 seeds
    python amazon_polarity_sweep.py --seeds 10 20  # subset
    python amazon_polarity_sweep.py --set bottom10 # contrast set
=============================================================================
"""

import os
import sys
import csv
import time
import argparse
import subprocess
import importlib.util
from collections import Counter

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# pipeline.py is the single source of truth for model + training.
_spec = importlib.util.spec_from_file_location(
    "rs", os.path.join(os.path.dirname(os.path.abspath(__file__)), "pipeline.py"))
rs = importlib.util.module_from_spec(_spec)
sys.modules["rs"] = rs
_spec.loader.exec_module(rs)

import torch

RAW_DIR   = os.path.join("cache", "amazon")
CACHE_DIR = os.path.join("cache", "amazon_encoded_leftpad")
OUT_DIR   = "amazon_polarity_results"
SEEDS     = [10, 20, 30, 40, 50]
ARCHS     = ["BiLSTM+LSTM", "BiGRU+GRU", "BiLSTM+BiLSTM", "BiGRU+BiGRU"]
FIELDS    = rs.FIELDS

N_TOTAL   = 50_000          # matches the IMDb corpus exactly
SPLIT_STATE = rs.SPLIT_STATE  # 42

# Top 10 pairs by mean test accuracy over all 4 architectures x 5 seeds
# in results/master_2000_runs.csv (the 2,000-run IMDb sweep).
TOP10 = [
    ("ReLU", "GELU"), ("ReLU", "ReLU"), ("ReLU", "Tanh"),
    ("LeakyReLU", "GELU"), ("GELU", "ELU"), ("LeakyReLU", "Mish"),
    ("PReLU", "ReLU"), ("LeakyReLU", "ReLU"), ("PReLU", "SiLU"),
    ("ReLU", "Hardswish"),
]

# Bottom 10 of the same ranking. Only used with --set bottom10; without a
# contrast set the top-10 numbers have nothing to be "top" relative to.
BOTTOM10 = [
    ("Hardswish", "LeakyReLU"), ("Tanh", "Tanh"), ("SELU", "SELU"),
    ("Tanh", "ReLU"), ("LeakyReLU", "SELU"), ("Hardswish", "PReLU"),
    ("Tanh", "SELU"), ("Mish", "SELU"), ("ELU", "LeakyReLU"),
    ("SiLU", "PReLU"),
]

PAIR_SETS = {"top10": TOP10, "bottom10": BOTTOM10}


# ─────────────────────────────────────────────────────────────────────────────
# DATA  -- rs.load_data() reproduced step for step on the Amazon corpus
# ─────────────────────────────────────────────────────────────────────────────
def build_cache():
    """Subsample, clean, split, build vocab, encode. Cached to .npy."""
    paths = {k: os.path.join(CACHE_DIR, f"{k}.npy")
             for k in ("X_train", "y_train", "X_test", "y_test")}
    if all(os.path.exists(p) for p in paths.values()):
        print(f"  cache hit: {CACHE_DIR}")
        return paths

    src = os.path.join(RAW_DIR, "raw_train.parquet")
    if not os.path.exists(src):
        raise SystemExit(f"\nERROR: {src} not found. Run download_amazon.py first.\n")

    print("\n[1/4] Loading Amazon Polarity ...")
    df = pd.read_parquet(src)
    print(f"  full corpus: {len(df):,} rows  labels={df.label.value_counts().to_dict()}")

    # Stratified subsample to IMDb's exact size, before any cleaning.
    from sklearn.model_selection import train_test_split
    df, _ = train_test_split(df, train_size=N_TOTAL, random_state=SPLIT_STATE,
                             stratify=df["label"].values)
    df = df.reset_index(drop=True)
    print(f"  subsampled : {len(df):,} rows  labels={df.label.value_counts().to_dict()}")

    print("[2/4] Cleaning (same clean_text as the IMDb pipeline) ...")
    # pandas 3 backs string columns with Arrow; sklearn cannot fancy-index
    # those, so force plain numpy object/int arrays as in the IMDb pipeline.
    texts  = np.asarray(df["text"].astype(str).apply(rs.clean_text).tolist(),
                        dtype=object)
    labels = np.asarray(df["label"].astype(int).tolist(), dtype=np.int64)

    print("[3/4] Splitting 80/20, stratified, random_state=42 ...")
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        texts, labels, test_size=0.20, random_state=SPLIT_STATE, stratify=labels)

    print("[4/4] Building vocab (5,000 most frequent train tokens) + encoding ...")
    counter = Counter()
    for t in X_train_text:
        counter.update(t.split())
    word2idx = {"<PAD>": 0, "<UNK>": 1}
    for w, _ in counter.most_common(rs.VOCAB_SIZE - 2):
        word2idx[w] = len(word2idx)

    def encode(ts):
        """LEFT-padded, unlike the IMDb pipeline's post-padding. See below.

        Both architectures read the FINAL hidden state (h[-1]). The IMDb
        pipeline post-pads, which is harmless there only because IMDb reviews
        are long: median 176 of 200 slots filled, so the readout sits ~24
        steps past the text. Amazon reviews are short (median 71), leaving
        ~129 steps of zero-embedding input before the readout, by which point
        the state has decayed and carries no signal. Measured: post-padding
        pins every run to 49.8% with loss frozen at 0.6931 = ln 2, while
        left-padding reaches 90.8% under the identical model and settings
        (diagnose_amazon.py).

        Left-padding changes no hyperparameter in Table III -- MAX_LEN is
        still 200, the vocabulary, truncation and everything downstream are
        untouched; only the side the zeros sit on moves. It is also what
        Keras pad_sequences does by default (padding='pre'), which is the
        lineage the original pipeline was transcribed from, so post-padding
        appears to be an artifact of the PyTorch port rather than a choice.
        """
        unk = word2idx["<UNK>"]
        out = np.zeros((len(ts), rs.MAX_LEN), dtype=np.int64)
        for i, text in enumerate(ts):
            ids = [word2idx.get(w, unk) for w in text.split()][:rs.MAX_LEN]
            if ids:
                out[i, rs.MAX_LEN - len(ids):] = ids
        return out

    X_train, X_test = encode(X_train_text), encode(X_test_text)
    lens = [min(len(t.split()), rs.MAX_LEN) for t in X_train_text[:5000]]
    cov  = sum(counter[w] for w in list(word2idx)[2:]) / max(sum(counter.values()), 1)
    print(f"  Train: {X_train.shape}  |  Test: {X_test.shape}")
    print(f"  median tokens/review (train, capped at {rs.MAX_LEN}): {int(np.median(lens))}")
    print(f"  vocab token coverage: {cov*100:.1f}% of train tokens")

    os.makedirs(CACHE_DIR, exist_ok=True)
    for k, v in zip(paths, (X_train, y_train, X_test, y_test)):
        np.save(paths[k], v)
    print(f"  cache written: {CACHE_DIR}")
    return paths


def load_cache():
    p = lambda k: os.path.join(CACHE_DIR, f"{k}.npy")
    return (np.load(p("X_train")), np.load(p("y_train")),
            np.load(p("X_test")),  np.load(p("y_test")))


# ─────────────────────────────────────────────────────────────────────────────
# WORKER  -- one seed, all pairs x all architectures
# ─────────────────────────────────────────────────────────────────────────────
def worker(seed, pairs, tag):
    out = os.path.join(OUT_DIR, f"{tag}_seed_{seed}.csv")
    os.makedirs(OUT_DIR, exist_ok=True)

    done = set()
    if os.path.exists(out):
        with open(out, newline="") as fh:
            for r in csv.DictReader(fh):
                done.add((r["act1"], r["act2"], r["experiment"]))

    todo = [(a, b, arch) for arch in ARCHS for (a, b) in pairs
            if (a, b, arch) not in done]
    if not todo:
        print(f"[seed {seed}] already complete ({len(done)} runs)")
        return

    X_tr, y_tr, X_te, y_te = load_cache()
    new_file = not os.path.exists(out)
    t0 = time.time()

    with open(out, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if new_file:
            w.writeheader()
        for i, (a1, a2, arch) in enumerate(todo, 1):
            m = rs.run_one(rs.MODEL_FOR[arch], a1, a2, seed,
                           X_tr, y_tr, X_te, y_te)
            w.writerow(dict(act1=a1, act2=a2, experiment=arch, seed=seed, **m))
            fh.flush()
            os.fsync(fh.fileno())
            eta = (time.time() - t0) / i * (len(todo) - i)
            print(f"[seed {seed}] {i:3d}/{len(todo)}  {arch:<14} "
                  f"{a1:>9}+{a2:<9} acc={m['accuracy']:6.2f}  "
                  f"ETA {eta/60:5.1f}m", flush=True)

    print(f"[seed {seed}] DONE in {(time.time()-t0)/60:.1f} min -> {out}", flush=True)


def merge(tag, seeds):
    frames = []
    for s in seeds:
        p = os.path.join(OUT_DIR, f"{tag}_seed_{s}.csv")
        if os.path.exists(p):
            frames.append(pd.read_csv(p))
    if not frames:
        return None
    d = pd.concat(frames, ignore_index=True)
    d["cell"] = d.act1 + "+" + d.act2
    p = os.path.join(OUT_DIR, f"{tag}_all_runs.csv")
    d.to_csv(p, index=False)
    print(f"\nmerged {len(d)} runs -> {p}")
    return d


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepare", action="store_true",
                    help="build the encoded cache and exit")
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--set", dest="pset", default="top10",
                    choices=list(PAIR_SETS))
    ap.add_argument("--worker", type=int, default=None,
                    help="internal: run one seed in this process")
    args = ap.parse_args()

    pairs = PAIR_SETS[args.pset]

    if args.prepare:
        build_cache()
        sys.exit(0)

    if args.worker is not None:
        worker(args.worker, pairs, args.pset)
        sys.exit(0)

    build_cache()
    n = len(pairs) * len(ARCHS) * len(args.seeds)
    print(f"\n{len(pairs)} pairs x {len(ARCHS)} architectures x "
          f"{len(args.seeds)} seeds = {n} runs")
    print(f"launching {len(args.seeds)} workers, one per seed\n")

    t0 = time.time()
    procs = [subprocess.Popen(
        [sys.executable, os.path.abspath(__file__), "--worker", str(s),
         "--set", args.pset]) for s in args.seeds]
    for p in procs:
        p.wait()

    print(f"\nall workers finished in {(time.time()-t0)/60:.1f} min")
    merge(args.pset, args.seeds)

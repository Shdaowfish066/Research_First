"""
=============================================================================
IMDb TOP-10 ACTIVATION PAIRS ON AMAZON POLARITY -- RIGHT/POST-PADDING
=============================================================================
Runs the ten activation pairs that lead the 2,000-run IMDb sweep on Amazon
Polarity under the IMDb pipeline's OWN padding convention: keep the first
200 tokens, then right-pad with zeros.

    10 pairs  x  4 architectures  x  5 seeds  =  200 runs

-----------------------------------------------------------------------------
READ THIS BEFORE INTERPRETING THE NUMBERS
-----------------------------------------------------------------------------
Right-padding is expected to produce near-chance accuracy on this corpus,
and that is the point of the run, not a defect in it.

Both architectures classify from the FINAL hidden state (h[-1]), and
encode() right-pads to MAX_LEN=200. IMDb reviews are long -- median 176 of
200 slots filled -- so the readout sits ~24 steps past the text. Amazon
Polarity reviews are short -- median 71 -- leaving ~129 steps of
zero-embedding input, over which the recurrent state contracts toward an
input-independent fixed point. A prior controlled measurement
(diagnose_amazon.py) recorded, for BiGRU+GRU / ReLU+ReLU / seed 10:

    Amazon, right-pad 200   49.83%   val loss 0.6931 = ln 2
    Amazon, left-pad  200   90.75%
    Amazon, right-pad 100   90.20%
    IMDb,   right-pad 200   86.72%   (control)

This script therefore EXPECTS results at or near 50.00% with loss ~0.6931.
Such results are recorded and reported as measured. Padding is never
switched, settings are never tuned, and no run is discarded for being
near chance. amazon_polarity_sweep.py is the left-padded companion; this
file does not modify it or its outputs.

-----------------------------------------------------------------------------
WHAT IS HELD IDENTICAL TO THE IMDb SWEEP
-----------------------------------------------------------------------------
The architectures, activation registry, set_seed(), make_loaders(), the
train/eval loop, optimiser, scheduler, dropout, gradient clipping, the
early-stopping rule and the metric computation are all IMPORTED from
pipeline.py rather than restated, so they cannot drift. run_one() calls
set_seed(seed) immediately before the model and its loaders are built.

No packing, no masking, no pooling change, no architectural change.

-----------------------------------------------------------------------------
DATA
-----------------------------------------------------------------------------
The same 50,000-review stratified subsample as amazon_polarity_sweep.py
(random_state=42), the same clean_text, the same 80/20 stratified split at
random_state=42, the same 5,000-token training vocabulary, PAD=0, UNK=1,
MAX_LEN=200. The ONLY difference from that script is the padding side.

The cache is rebuilt from the raw parquet into its own directory,
cache/amazon_test_top_10_rightpad/, and is then VERIFIED row by row against
the existing left-padded cache: identical labels, and identical nonzero
token sequences in identical order. That check is what proves no
left-padded array has been reused and that split membership is preserved.

Usage:
    python amazon_test_top_10.py --prepare    # build + verify the cache
    python amazon_test_top_10.py              # run all 200
    python amazon_test_top_10.py --analyze    # report (needs all 200)
=============================================================================
"""

import os
import sys
import csv
import json
import time
import hashlib
import argparse
import subprocess
import importlib.util
from collections import Counter

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, fname))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# amazon_polarity_sweep imports pipeline.py as "rs"; reuse both so the pair
# list and the training code have exactly one definition each.
aps = _load("aps", "amazon_polarity_sweep.py")
rs = aps.rs

import torch

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
PADDING     = "right"                                   # the whole point
RAW_DIR     = os.path.join("cache", "amazon")
CACHE_DIR   = os.path.join("cache", "amazon_test_top_10_rightpad")
LEFTPAD_DIR = os.path.join("cache", "amazon_encoded_leftpad")  # read-only check
OUT_DIR     = "amazon_test_top_10_results"
MANIFEST    = os.path.join(OUT_DIR, "run_config.json")

SEEDS       = [10, 20, 30, 40, 50]
ARCHS       = ["BiLSTM+LSTM", "BiGRU+GRU", "BiLSTM+BiLSTM", "BiGRU+BiGRU"]
TOP10       = list(aps.TOP10)          # NOT re-selected or re-ranked here
N_TOTAL     = aps.N_TOTAL              # 50,000
SPLIT_STATE = rs.SPLIT_STATE           # 42
FIELDS      = rs.FIELDS
EXPECTED    = len(TOP10) * len(ARCHS) * len(SEEDS)      # 200

IMDB_MASTER = os.path.join("results", "master_2000_runs.csv")

# Outcomes under right-padding on this corpus are bimodal, not uniformly at
# chance: most runs sit exactly at 50.00 with precision=recall=0 (one class
# predicted for everything), while an occasional seed escapes the plateau and
# reaches ~89%. These thresholds only CLASSIFY runs for reporting -- no run is
# ever filtered, reweighted or excluded by them.
CHANCE_MAX = 55.0     # at/near chance
ESCAPE_MIN = 80.0     # clearly learned


def fingerprint(*arrays):
    h = hashlib.sha256()
    for a in arrays:
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()[:16]


def config_dict(cache_fp):
    return {
        "experiment": "IMDb top-10 activation pairs on Amazon Polarity",
        "padding": PADDING,
        "pairs": [f"{a}+{b}" for a, b in TOP10],
        "pair_source": "amazon_polarity_sweep.TOP10 (imported, not re-ranked)",
        "pair_provenance": "top 10 by mean test accuracy over 4 archs x 5 "
                           "seeds in results/master_2000_runs.csv",
        "architectures": ARCHS,
        "seeds": SEEDS,
        "expected_runs": EXPECTED,
        "corpus": "mteb/amazon_polarity @ e2d317d38cd51312af73b3d32a06d1a08b442046",
        "subsample": N_TOTAL,
        "split_state": SPLIT_STATE,
        "vocab_size": rs.VOCAB_SIZE,
        "max_len": rs.MAX_LEN,
        "pad_idx": 0,
        "unk_idx": 1,
        "batch_size": rs.BATCH_SIZE,
        "epochs": rs.EPOCHS,
        "learning_rate": rs.LEARNING_RATE,
        "dropout": rs.DROPOUT_RATE,
        "patience_stop": rs.PATIENCE_STOP,
        "val_split": rs.VAL_SPLIT,
        "cache_dir": CACHE_DIR,
        "cache_fingerprint": cache_fp,
        "expectation": "near-chance (~50%, loss ~0.6931) is the predicted "
                       "outcome under right-padding on this corpus and is "
                       "reported as measured, not corrected",
    }


# ─────────────────────────────────────────────────────────────────────────────
# DATA  -- rs.load_data() reproduced, RIGHT-padded
# ─────────────────────────────────────────────────────────────────────────────
def _paths():
    return {k: os.path.join(CACHE_DIR, f"{k}.npy")
            for k in ("X_train", "y_train", "X_test", "y_test")}


def build_cache():
    paths = _paths()
    if all(os.path.exists(p) for p in paths.values()):
        print(f"  cache hit: {CACHE_DIR}")
        return

    src = os.path.join(RAW_DIR, "raw_train.parquet")
    if not os.path.exists(src):
        raise SystemExit(
            f"\nBLOCKER: {src} not found.\n"
            f"Run:  python download_amazon.py\n")

    from sklearn.model_selection import train_test_split

    print("\n[1/4] Loading Amazon Polarity ...")
    df = pd.read_parquet(src)
    print(f"  full corpus: {len(df):,} rows")

    # Identical subsample to amazon_polarity_sweep.build_cache().
    df, _ = train_test_split(df, train_size=N_TOTAL, random_state=SPLIT_STATE,
                             stratify=df["label"].values)
    df = df.reset_index(drop=True)
    print(f"  subsampled : {len(df):,} rows  "
          f"labels={df.label.value_counts().to_dict()}")

    print("[2/4] Cleaning (pipeline clean_text) ...")
    texts = np.asarray(df["text"].astype(str).apply(rs.clean_text).tolist(),
                       dtype=object)
    labels = np.asarray(df["label"].astype(int).tolist(), dtype=np.int64)

    print(f"[3/4] Splitting 80/20, stratified, random_state={SPLIT_STATE} ...")
    Xtr_t, Xte_t, y_train, y_test = train_test_split(
        texts, labels, test_size=0.20, random_state=SPLIT_STATE, stratify=labels)

    print(f"[4/4] Vocab ({rs.VOCAB_SIZE:,}) + RIGHT-padded encoding ...")
    counter = Counter()
    for t in Xtr_t:
        counter.update(t.split())
    word2idx = {"<PAD>": 0, "<UNK>": 1}
    for w, _ in counter.most_common(rs.VOCAB_SIZE - 2):
        word2idx[w] = len(word2idx)

    def encode(ts):
        """First MAX_LEN tokens, then RIGHT-pad -- verbatim rs.load_data()."""
        unk = word2idx["<UNK>"]
        out = np.zeros((len(ts), rs.MAX_LEN), dtype=np.int64)
        for i, text in enumerate(ts):
            ids = [word2idx.get(w, unk) for w in text.split()][:rs.MAX_LEN]
            out[i, :len(ids)] = ids
        return out

    X_train, X_test = encode(Xtr_t), encode(Xte_t)
    print(f"  Train: {X_train.shape}  |  Test: {X_test.shape}")

    os.makedirs(CACHE_DIR, exist_ok=True)
    for k, v in zip(paths, (X_train, y_train, X_test, y_test)):
        np.save(paths[k], v)
    print(f"  cache written: {CACHE_DIR}")


def load_cache():
    p = _paths()
    return (np.load(p["X_train"]), np.load(p["y_train"]),
            np.load(p["X_test"]),  np.load(p["y_test"]))


def nonzero_seq(row):
    """The row's tokens with padding removed, order preserved."""
    nz = np.nonzero(row)[0]
    if len(nz) == 0:
        return row[:0]
    return row[nz[0]:nz[-1] + 1]


def verify_cache():
    """Padding placement, ids, shapes, empty rows, and provenance vs left-pad."""
    X_tr, y_tr, X_te, y_te = load_cache()
    ok = True
    print("\n" + "=" * 74)
    print("CACHE VERIFICATION")
    print("=" * 74)

    # -- shapes ------------------------------------------------------------
    n_tr, n_te = int(N_TOTAL * 0.8), int(N_TOTAL * 0.2)
    for nm, X, y, n in (("train", X_tr, y_tr, n_tr), ("test", X_te, y_te, n_te)):
        good = X.shape == (n, rs.MAX_LEN) and y.shape == (n,)
        ok &= good
        print(f"  shape {nm:<5} X{X.shape} y{y.shape}  "
              f"expect ({n}, {rs.MAX_LEN}) / ({n},)   "
              f"{'OK' if good else 'FAIL'}")

    # -- dtype and id range ------------------------------------------------
    for nm, X in (("train", X_tr), ("test", X_te)):
        good = (X.dtype == np.int64 and X.min() >= 0 and X.max() < rs.VOCAB_SIZE)
        ok &= good
        print(f"  ids   {nm:<5} dtype={X.dtype} min={X.min()} max={X.max()}  "
              f"expect [0, {rs.VOCAB_SIZE})   {'OK' if good else 'FAIL'}")

    # -- padding placement: zeros must be a SUFFIX, never a prefix ---------
    for nm, X in (("train", X_tr), ("test", X_te)):
        short = X[(X == 0).any(axis=1) & (X != 0).any(axis=1)]
        lead_zero = int((short[:, 0] == 0).sum())          # must be 0
        trail_zero = int((short[:, -1] == 0).sum())        # must be all
        # a right-padded row is nonzero up to L then zero for the rest
        L = (X != 0).sum(axis=1)
        idx = np.arange(rs.MAX_LEN)[None, :]
        contiguous = bool(((idx < L[:, None]) == (X != 0)).all())
        good = lead_zero == 0 and trail_zero == len(short) and contiguous
        ok &= good
        print(f"  pad   {nm:<5} padded rows={len(short):<6} "
              f"starting with 0: {lead_zero} (want 0)   "
              f"ending with 0: {trail_zero}/{len(short)}   "
              f"contiguous-prefix: {contiguous}   {'OK' if good else 'FAIL'}")

    # -- empty sequences ---------------------------------------------------
    for nm, X in (("train", X_tr), ("test", X_te)):
        empty = int((X != 0).sum(axis=1).eq(0).sum()) if hasattr(
            (X != 0).sum(axis=1), "eq") else int(((X != 0).sum(axis=1) == 0).sum())
        print(f"  empty {nm:<5} all-padding rows: {empty}"
              f"{'  (these carry no signal at all)' if empty else ''}")

    # -- labels ------------------------------------------------------------
    for nm, y in (("train", y_tr), ("test", y_te)):
        good = set(np.unique(y)) == {0, 1}
        ok &= good
        print(f"  label {nm:<5} classes={sorted(set(np.unique(y).tolist()))} "
              f"pos_frac={y.mean():.4f}   {'OK' if good else 'FAIL'}")

    # -- provenance: same rows as the left-padded cache, re-padded ---------
    lp = {k: os.path.join(LEFTPAD_DIR, f"{k}.npy")
          for k in ("X_train", "y_train", "X_test", "y_test")}
    if all(os.path.exists(v) for v in lp.values()):
        print("\n  cross-check against the left-padded cache "
              "(must be the same data, re-padded):")
        for nm, Xr, yr, kx, ky in (("train", X_tr, y_tr, "X_train", "y_train"),
                                   ("test",  X_te, y_te, "X_test",  "y_test")):
            Xl, yl = np.load(lp[kx]), np.load(lp[ky])
            lab_same = bool((yr == yl).all())
            m = min(2000, len(Xr))
            tok_same = all(np.array_equal(nonzero_seq(Xr[i]), nonzero_seq(Xl[i]))
                           for i in range(m))
            len_same = bool(((Xr != 0).sum(1) == (Xl != 0).sum(1)).all())
            differs = not np.array_equal(Xr, Xl)
            ok &= lab_same and tok_same and len_same and differs
            print(f"    {nm:<5} labels identical: {lab_same}   "
                  f"token sequences identical (first {m}): {tok_same}   "
                  f"lengths identical: {len_same}   "
                  f"arrays differ (i.e. not reused): {differs}")
    else:
        print("\n  left-padded cache absent; skipping cross-check "
              "(cache was built from the raw parquet, so provenance holds)")

    print("=" * 74)
    print(f"VERIFICATION {'PASSED' if ok else 'FAILED'}")
    print("=" * 74)
    if not ok:
        raise SystemExit("\nBLOCKER: cache verification failed; not training.\n")
    return fingerprint(X_tr, y_tr, X_te, y_te)


# ─────────────────────────────────────────────────────────────────────────────
# MANIFEST  -- reject incompatible cache or resumed results
# ─────────────────────────────────────────────────────────────────────────────
def check_manifest(cache_fp):
    os.makedirs(OUT_DIR, exist_ok=True)
    cur = config_dict(cache_fp)
    if os.path.exists(MANIFEST):
        old = json.load(open(MANIFEST, encoding="utf-8"))
        keys = ["padding", "pairs", "architectures", "seeds", "subsample",
                "split_state", "vocab_size", "max_len", "batch_size", "epochs",
                "learning_rate", "dropout", "patience_stop", "val_split",
                "cache_fingerprint"]
        diff = {k: (old.get(k), cur.get(k)) for k in keys
                if old.get(k) != cur.get(k)}
        if diff:
            msg = "\n".join(f"    {k}: existing={o!r}  now={n!r}"
                            for k, (o, n) in diff.items())
            raise SystemExit(
                f"\nBLOCKER: {OUT_DIR}/ holds results from an incompatible "
                f"configuration.\n{msg}\n\n"
                f"  Resuming would mix them. Move or delete {OUT_DIR}/ to "
                f"start clean.\n")
        print(f"  manifest OK (cache {cache_fp}); resuming is safe")
    else:
        json.dump(cur, open(MANIFEST, "w", encoding="utf-8"), indent=2)
        print(f"  manifest written: {MANIFEST}")


# ─────────────────────────────────────────────────────────────────────────────
# WORKER  -- one seed, 10 pairs x 4 architectures = 40 runs
# ─────────────────────────────────────────────────────────────────────────────
def worker(seed):
    out = os.path.join(OUT_DIR, f"seed_{seed}.csv")
    os.makedirs(OUT_DIR, exist_ok=True)

    done = set()
    if os.path.exists(out):
        with open(out, newline="") as fh:
            for r in csv.DictReader(fh):
                done.add((r["act1"], r["act2"], r["experiment"]))

    todo = [(a, b, arch) for arch in ARCHS for (a, b) in TOP10
            if (a, b, arch) not in done]
    if not todo:
        print(f"[seed {seed}] already complete ({len(done)}/40)")
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
            os.fsync(fh.fileno())           # each run durable immediately
            eta = (time.time() - t0) / i * (len(todo) - i)
            print(f"[seed {seed}] {i:2d}/{len(todo)}  {arch:<14} "
                  f"{a1:>9}+{a2:<9} acc={m['accuracy']:6.2f}  "
                  f"ETA {eta/60:5.1f}m", flush=True)

    print(f"[seed {seed}] DONE in {(time.time()-t0)/60:.1f} min -> {out}",
          flush=True)


def merge():
    frames = []
    for s in SEEDS:
        p = os.path.join(OUT_DIR, f"seed_{s}.csv")
        if os.path.exists(p):
            frames.append(pd.read_csv(p))
    if not frames:
        return None
    d = pd.concat(frames, ignore_index=True)
    d["cell"] = d.act1 + "+" + d.act2
    before = len(d)
    d = d.drop_duplicates(subset=["cell", "experiment", "seed"], keep="first")
    if len(d) != before:
        print(f"  dropped {before - len(d)} duplicate rows on merge")
    p = os.path.join(OUT_DIR, "all_runs.csv")
    d.to_csv(p, index=False)
    print(f"merged {len(d)} runs -> {p}")
    return d


# ─────────────────────────────────────────────────────────────────────────────
# ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
def analyze():
    from scipy.stats import spearmanr

    p = os.path.join(OUT_DIR, "all_runs.csv")
    d = merge() if not os.path.exists(p) else pd.read_csv(p)
    if d is None or not len(d):
        raise SystemExit(f"\nBLOCKER: no results in {OUT_DIR}/.\n")
    if "cell" not in d.columns:
        d["cell"] = d.act1 + "+" + d.act2

    W = 96
    print("=" * W)
    print("COMPLETENESS")
    print("=" * W)
    combos = d.groupby(["cell", "experiment"]).seed.nunique()
    uniq = len(d.drop_duplicates(subset=["cell", "experiment", "seed"]))
    complete = uniq == EXPECTED and bool((combos == len(SEEDS)).all())
    print(f"  unique (pair, architecture, seed) runs : {uniq} / {EXPECTED}")
    print(f"  pairs {d.cell.nunique()}/10   architectures "
          f"{d.experiment.nunique()}/4   seeds {sorted(d.seed.unique())}")
    print(f"  every (pair, arch) has all 5 seeds     : "
          f"{bool((combos == len(SEEDS)).all())}")
    print(f"  STATUS: {'COMPLETE' if complete else 'INCOMPLETE'}")
    if not complete:
        missing = [(c, a) for c in d.cell.unique() for a in ARCHS
                   if combos.get((c, a), 0) != len(SEEDS)]
        print(f"  missing/partial (pair, arch) cells: {len(missing)}")
        for c, a in missing[:12]:
            print(f"    {c:<20} {a:<14} seeds={combos.get((c, a), 0)}/5")
    print()

    print("=" * W)
    print(f"PER-ARCHITECTURE RESULTS  (mean +/- sample SD over "
          f"{len(SEEDS)} seeds; padding={PADDING})")
    print("=" * W)
    for arch in ARCHS:
        g = d[d.experiment == arch]
        if not len(g):
            continue
        agg = g.groupby("cell").agg(
            acc=("accuracy", "mean"), acc_sd=("accuracy", "std"),
            pre=("precision", "mean"), pre_sd=("precision", "std"),
            rec=("recall", "mean"), rec_sd=("recall", "std"),
            f1=("f1", "mean"), f1_sd=("f1", "std"),
            n=("accuracy", "count")).sort_values("acc", ascending=False)
        esc = g[g.accuracy >= ESCAPE_MIN].groupby("cell").size()
        chn = g[g.accuracy <= CHANCE_MAX].groupby("cell").size()
        print(f"\n  {arch}   (rank is WITHIN these 10 pairs, not among all 100)")
        print(f"  {'rank':<5}{'pair':<20}{'accuracy':>18}{'precision':>18}"
              f"{'recall':>18}{'F1':>18}{'n':>4}{'esc':>5}{'chance':>7}")
        for i, (c, r) in enumerate(agg.iterrows(), 1):
            print(f"  {i:<5}{c:<20}"
                  f"{r.acc:>9.2f}+/-{r.acc_sd:<6.2f}"
                  f"{r.pre:>9.2f}+/-{r.pre_sd:<6.2f}"
                  f"{r.rec:>9.2f}+/-{r.rec_sd:<6.2f}"
                  f"{r.f1:>9.2f}+/-{r.f1_sd:<6.2f}{int(r.n):>4}"
                  f"{int(esc.get(c, 0)):>5}{int(chn.get(c, 0)):>7}")
        print(f"    'esc' = seeds reaching >={ESCAPE_MIN:.0f}%, "
              f"'chance' = seeds at <={CHANCE_MAX:.0f}%, out of {len(SEEDS)}")

    print()
    print("=" * W)
    print("OVERALL RANKING  (accuracy averaged equally over the 4 "
          "architectures and 5 seeds)")
    print("=" * W)
    arch_mean = d.groupby(["cell", "experiment"]).accuracy.mean()
    overall = arch_mean.groupby("cell").mean().sort_values(ascending=False)
    overall_sd = d.groupby("cell").accuracy.std()
    print(f"  {'rank':<5}{'pair':<20}{'accuracy':>10}{'SD(20 runs)':>14}")
    for i, (c, v) in enumerate(overall.items(), 1):
        print(f"  {i:<5}{c:<20}{v:>10.2f}{overall_sd[c]:>14.2f}")
    print(f"\n  span of the ten: "
          f"{overall.max() - overall.min():.2f} points")
    print("  NOTE: this ranks the 10 selected pairs against each other only.")

    # Bimodality, not the mean, is what invalidates the ranking here: a pair
    # with one escaped seed and four dead ones averages ~58%, which no
    # distance-from-50 test would flag, yet its mean encodes only how many
    # seeds happened to escape.
    n_chance = int((d.accuracy <= CHANCE_MAX).sum())
    n_esc = int((d.accuracy >= ESCAPE_MIN).sum())
    n_mid = len(d) - n_chance - n_esc
    print(f"\n  run outcomes: {n_chance}/{len(d)} at chance (<={CHANCE_MAX:.0f}%), "
          f"{n_esc}/{len(d)} learned (>={ESCAPE_MIN:.0f}%), {n_mid}/{len(d)} between")
    degenerate = d.accuracy.eq(50.0).sum()
    print(f"  runs at exactly 50.00% (one class predicted for every input): "
          f"{degenerate}/{len(d)}")

    if n_chance >= 0.25 * len(d):
        print(f"\n  *** {n_chance/len(d)*100:.0f}% of runs never left the chance "
              f"plateau. Under right-padding on this\n      corpus that is the "
              f"predicted outcome (see the module docstring): the\n      readout "
              f"sits ~129 steps past the text. Outcomes are bimodal -- a run\n"
              f"      either escapes or does not -- so a pair's mean encodes "
              f"mainly HOW MANY\n      of its five seeds escaped, and the SD is "
              f"driven by that split rather\n      than by activation behaviour. "
              f"The ranking above must NOT be read as a\n      preference "
              f"ordering over activation pairs.")

    print()
    print("=" * W)
    print("COMPARISON WITH IMDb  (same 10 pairs, same aggregation)")
    print("=" * W)
    if not os.path.exists(IMDB_MASTER):
        print(f"  {IMDB_MASTER} not found; skipping.")
    else:
        im = pd.read_csv(IMDB_MASTER)
        im = im[im.cell.isin(overall.index)]
        im_arch = im.groupby(["cell", "experiment"]).accuracy.mean()
        im_overall = im_arch.groupby("cell").mean()
        both = pd.DataFrame({"imdb": im_overall, "amazon": overall}).dropna()
        both["imdb_rank"] = both.imdb.rank(ascending=False)
        both["amazon_rank"] = both.amazon.rank(ascending=False)
        both = both.sort_values("imdb_rank")
        print(f"  {'pair':<20}{'IMDb %':>9}{'rank':>6}"
              f"{'Amazon %':>11}{'rank':>6}{'d rank':>8}{'d acc':>9}")
        for c, r in both.iterrows():
            print(f"  {c:<20}{r.imdb:>9.2f}{int(r.imdb_rank):>6}"
                  f"{r.amazon:>11.2f}{int(r.amazon_rank):>6}"
                  f"{int(r.imdb_rank - r.amazon_rank):>+8d}"
                  f"{r.amazon - r.imdb:>+9.2f}")
        rho, pv = spearmanr(both.imdb, both.amazon)
        print("-" * W)
        print(f"  Spearman rho (within these 10 pairs) = {rho:+.3f}   p = {pv:.4g}")
        shift = (both.imdb_rank - both.amazon_rank).abs().mean()
        print(f"  mean |rank shift| = {shift:.1f} places")
        print(f"  accuracy level: IMDb {both.imdb.mean():.2f}%  ->  "
              f"Amazon {both.amazon.mean():.2f}%  "
              f"({both.amazon.mean() - both.imdb.mean():+.2f} pts)")
        print("  NOTE: correlation is over the 10 selected pairs only, so it "
              "describes\n        re-ordering within a pre-filtered set, not "
              "agreement across all 100.")
        both.round(4).to_csv(
            os.path.join(OUT_DIR, "imdb_vs_amazon_top10.csv"))

    d.groupby(["experiment", "cell"]).agg(
        acc=("accuracy", "mean"), acc_sd=("accuracy", "std"),
        pre=("precision", "mean"), pre_sd=("precision", "std"),
        rec=("recall", "mean"), rec_sd=("recall", "std"),
        f1=("f1", "mean"), f1_sd=("f1", "std"),
        n=("accuracy", "count")).round(4).to_csv(
        os.path.join(OUT_DIR, "summary_by_pair_arch.csv"))
    overall.round(4).to_frame("accuracy").to_csv(
        os.path.join(OUT_DIR, "summary_overall.csv"))
    print(f"\nwrote {OUT_DIR}/summary_by_pair_arch.csv, "
          f"{OUT_DIR}/summary_overall.csv")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--worker", type=int, default=None)
    args = ap.parse_args()

    if args.analyze:
        analyze()
        sys.exit(0)

    if args.worker is not None:
        worker(args.worker)
        sys.exit(0)

    build_cache()
    fp = verify_cache()
    if args.prepare:
        print(f"\ncache fingerprint {fp}; ready.")
        sys.exit(0)

    check_manifest(fp)

    if rs.DEVICE.type != "cuda":
        raise SystemExit(
            "\nBLOCKER: no CUDA device. This is 200 recurrent training runs; "
            "CPU is not a\nviable fallback. Resolve GPU access, then:\n"
            "    python amazon_test_top_10.py\n")

    print(f"\n{len(TOP10)} pairs x {len(ARCHS)} architectures x "
          f"{len(SEEDS)} seeds = {EXPECTED} runs")
    print(f"padding = {PADDING}   device = {torch.cuda.get_device_name(0)}")
    print(f"launching {len(SEEDS)} workers, one per seed\n")

    t0 = time.time()
    procs = [subprocess.Popen(
        [sys.executable, os.path.abspath(__file__), "--worker", str(s)])
        for s in SEEDS]
    for p in procs:
        p.wait()

    print(f"\nall workers finished in {(time.time()-t0)/60:.1f} min")
    d = merge()
    if d is not None:
        uniq = len(d.drop_duplicates(subset=["cell", "experiment", "seed"]))
        print(f"completed {uniq}/{EXPECTED} runs")
        if uniq == EXPECTED:
            print("\n")
            analyze()
        else:
            print("\nIncomplete. Re-run the same command to resume.")

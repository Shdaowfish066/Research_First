"""
=============================================================================
FULL 5-SEED SWEEP OF ONE ARCHITECTURE  (iCONEECT 2026)
=============================================================================
Runs all 100 activation pairs (10 Act1 x 10 Act2) of ONE architecture under
5 seeds = 500 training runs, so every cell gets a mean +/- std instead of a
single noisy draw.

    100 cells (configurations)  x  5 seeds  =  500 runs

Output tree (one folder per architecture):

    <ARCH>_5_seed_variance_results/
        seed_10.csv          100 rows -- every pair at seed 10
        seed_20.csv          100 rows -- every pair at seed 20
        seed_30.csv          ...
        seed_40.csv
        seed_50.csv
        all_runs.csv         500 rows -- the five above, merged
        mean_100_cells.csv   100 rows -- mean +/- std +/- 95% CI per cell

-----------------------------------------------------------------------------
PARALLELISM  --  why this does not change any number
-----------------------------------------------------------------------------
These models are tiny (~764K params, batch 64), so a single run leaves the
GPU mostly idle -- the cost is kernel-launch overhead on a 200-step recurrence,
not arithmetic. We therefore run ONE WORKER PROCESS PER SEED, five at once.

Each worker is a separate OS process with its own CUDA context, its own RNG,
and its own output file. Runs never share state. Every worker calls set_seed()
immediately before building each model, exactly as pipeline.py does, so a
given (pair, seed) produces the same weights and the same batch order no
matter which worker runs it, in what order, or alongside what else.
cudnn.deterministic=True with benchmark=False also pins kernel selection, so
GPU load cannot change which algorithm is chosen.

Net effect: identical numbers to a serial run, roughly 4-5x faster wall clock.
A preflight check (--preflight) verifies this empirically before committing
hours of GPU time: it runs the same (pair, seed) in two concurrent processes
and asserts the metrics match to the last decimal.

-----------------------------------------------------------------------------
DATA CACHE
-----------------------------------------------------------------------------
Preprocessing (clean, split, build vocab, encode) is deterministic and takes
~40 s. Doing it in five workers would waste time and RAM, so the parent does
it once and caches the encoded arrays to .npy. Workers memory-map them. The
split stays pinned at random_state=42 exactly as in every other script here.

-----------------------------------------------------------------------------
Pipeline (preprocessing, architecture, Adam 1e-3, ReduceLROnPlateau,
grad-clip 1.0, early stop patience 3) is inherited from pipeline.py, which
was transcribed verbatim from the original sweep scripts.

Resumable: each worker skips (pair, seed) rows already present in its own CSV,
and fsyncs after every run, so a crash or Ctrl-C loses at most one run.

Usage:
    python run_full_sweep.py --arch "BiLSTM+LSTM"
    python run_full_sweep.py --arch "BiLSTM+LSTM" --preflight
    python run_full_sweep.py --arch "BiGRU+GRU" --seeds 10 20 30 40 50
=============================================================================
"""

import os
import sys
import csv
import time
import argparse
import subprocess
import itertools

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# pipeline.py is the single source of truth for the pipeline.
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "rs", os.path.join(os.path.dirname(os.path.abspath(__file__)), "pipeline.py"))
rs = importlib.util.module_from_spec(_spec)
sys.modules["rs"] = rs
_spec.loader.exec_module(rs)

import torch
from scipy import stats

ACT_NAMES = list(rs.ACTIVATIONS.keys())          # the canonical order of the 10
DEFAULT_SEEDS = [10, 20, 30, 40, 50]
CACHE_DIR = os.path.join("cache", "sweep_data")
FIELDS = rs.FIELDS                               # act1..stop_epoch, same schema


def out_dir_for(arch):
    return f"{arch}_5_seed_variance_results"


# ─────────────────────────────────────────────────────────────────────────────
# DATA CACHE
# ─────────────────────────────────────────────────────────────────────────────
def build_cache():
    """Encode once in the parent; workers just np.load it."""
    paths = {k: os.path.join(CACHE_DIR, f"{k}.npy")
             for k in ("X_train", "y_train", "X_test", "y_test")}
    if all(os.path.exists(p) for p in paths.values()):
        print(f"  cache hit: {CACHE_DIR}")
        return paths
    os.makedirs(CACHE_DIR, exist_ok=True)
    X_tr, y_tr, X_te, y_te = rs.load_data()
    for k, v in zip(paths, (X_tr, y_tr, X_te, y_te)):
        np.save(paths[k], v)
    print(f"  cache written: {CACHE_DIR}")
    return paths


def load_cache():
    p = lambda k: os.path.join(CACHE_DIR, f"{k}.npy")
    return (np.load(p("X_train")), np.load(p("y_train")),
            np.load(p("X_test")),  np.load(p("y_test")))


# ─────────────────────────────────────────────────────────────────────────────
# WORKER  -- one seed, all 100 pairs
# ─────────────────────────────────────────────────────────────────────────────
def worker(arch, seed, quiet=False):
    out = os.path.join(out_dir_for(arch), f"seed_{seed}.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    done = set()
    if os.path.exists(out):
        with open(out, newline="") as fh:
            for r in csv.DictReader(fh):
                done.add((r["act1"], r["act2"]))

    pairs = [(a, b) for a in ACT_NAMES for b in ACT_NAMES]
    todo = [p for p in pairs if p not in done]
    if not todo:
        print(f"[seed {seed}] already complete ({len(done)}/100)")
        return

    X_tr, y_tr, X_te, y_te = load_cache()
    model_cls = rs.MODEL_FOR[arch]
    new_file = not os.path.exists(out)
    t0 = time.time()

    with open(out, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if new_file:
            w.writeheader()
        for i, (a1, a2) in enumerate(todo, 1):
            m = rs.run_one(model_cls, a1, a2, seed, X_tr, y_tr, X_te, y_te)
            w.writerow(dict(act1=a1, act2=a2, experiment=arch, seed=seed, **m))
            fh.flush()
            os.fsync(fh.fileno())
            if not quiet:
                eta = (time.time() - t0) / i * (len(todo) - i)
                print(f"[seed {seed}] {i:3d}/{len(todo)}  {a1:>9}+{a2:<9} "
                      f"acc={m['accuracy']:6.2f}  ETA {eta/60:5.1f}m", flush=True)

    print(f"[seed {seed}] DONE in {(time.time()-t0)/60:.1f} min -> {out}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# PREFLIGHT  -- prove concurrency does not perturb results
# ─────────────────────────────────────────────────────────────────────────────
def preflight(arch, seed):
    """Same (pair, seed) in two concurrent processes must agree exactly."""
    print(f"\nPREFLIGHT: running Tanh+Hardswish @ seed {seed} in 2 concurrent "
          f"processes\n(if concurrency perturbed anything, these would differ)\n")
    cmd = [sys.executable, os.path.abspath(__file__), "--arch", arch,
           "--single", "Tanh", "Hardswish", "--seed", str(seed)]
    procs = [subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True)
             for _ in range(2)]
    outs = [p.communicate()[0].strip().splitlines()[-1] for p in procs]
    for i, o in enumerate(outs, 1):
        print(f"  process {i}: {o}")
    ok = outs[0] == outs[1]
    print(f"\n  {'MATCH - concurrency is safe' if ok else 'MISMATCH - DO NOT RUN IN PARALLEL'}")
    return ok


def single(arch, a1, a2, seed):
    X_tr, y_tr, X_te, y_te = load_cache()
    m = rs.run_one(rs.MODEL_FOR[arch], a1, a2, seed, X_tr, y_tr, X_te, y_te)
    print(f"acc={m['accuracy']} prec={m['precision']} rec={m['recall']} "
          f"f1={m['f1']} best_ep={m['best_epoch']} stop_ep={m['stop_epoch']}")


# ─────────────────────────────────────────────────────────────────────────────
# MERGE + PER-CELL SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
def summarise(arch, seeds):
    d = out_dir_for(arch)
    frames = []
    for s in seeds:
        p = os.path.join(d, f"seed_{s}.csv")
        if os.path.exists(p):
            frames.append(pd.read_csv(p))
    if not frames:
        print("nothing to summarise")
        return
    allr = pd.concat(frames, ignore_index=True)
    allr.to_csv(os.path.join(d, "all_runs.csv"), index=False)

    rows = []
    for (a1, a2), g in allr.groupby(["act1", "act2"]):
        acc, f1 = g.accuracy.to_numpy(), g.f1.to_numpy()
        n = len(acc)
        ci = (stats.t.ppf(0.975, n - 1) * acc.std(ddof=1) / np.sqrt(n)
              if n > 1 else float("nan"))
        rows.append(dict(
            act1=a1, act2=a2, experiment=arch, n_seeds=n,
            acc_mean=round(acc.mean(), 3), acc_std=round(acc.std(ddof=1), 3) if n > 1 else np.nan,
            acc_ci95=round(ci, 3) if n > 1 else np.nan,
            acc_min=acc.min(), acc_max=acc.max(),
            f1_mean=round(f1.mean(), 3), f1_std=round(f1.std(ddof=1), 3) if n > 1 else np.nan,
            prec_mean=round(g.precision.mean(), 3), rec_mean=round(g.recall.mean(), 3),
        ))
    summ = pd.DataFrame(rows).sort_values("acc_mean", ascending=False)
    path = os.path.join(d, "mean_100_cells.csv")
    summ.to_csv(path, index=False)

    print(f"\n{'='*70}")
    print(f"{arch}: {len(allr)} runs over {len(frames)} seeds -> "
          f"{len(summ)} cells")
    print(f"{'='*70}")
    print(f"  all runs      -> {os.path.join(d,'all_runs.csv')}")
    print(f"  per-cell mean -> {path}")
    if len(summ) and summ.n_seeds.min() > 1:
        print(f"\n  per-cell std: {summ.acc_std.min():.2f} to "
              f"{summ.acc_std.max():.2f} (median {summ.acc_std.median():.2f})")
        print(f"\n  top 5 cells by mean accuracy:")
        for r in summ.head(5).itertuples():
            print(f"    {r.act1:>9} + {r.act2:<9} {r.acc_mean:6.2f} "
                  f"+/- {r.acc_std:.2f}")
    return summ


# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="BiLSTM+LSTM", choices=list(rs.MODEL_FOR))
    ap.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    ap.add_argument("--preflight", action="store_true")
    ap.add_argument("--summarise-only", action="store_true")
    # internal
    ap.add_argument("--worker", type=int)
    ap.add_argument("--single", nargs=2, metavar=("ACT1", "ACT2"))
    ap.add_argument("--seed", type=int)
    a = ap.parse_args()

    if a.single:
        return single(a.arch, a.single[0], a.single[1], a.seed)
    if a.worker is not None:
        return worker(a.arch, a.worker)
    if a.summarise_only:
        return summarise(a.arch, a.seeds)

    print("=" * 70)
    print(f"FULL 5-SEED SWEEP  --  {a.arch}")
    print(f"  100 cells x {len(a.seeds)} seeds {a.seeds} = "
          f"{100*len(a.seeds)} runs")
    print(f"  output: {out_dir_for(a.arch)}/")
    print("=" * 70)

    print("\n[1/4] Preparing data cache ...")
    build_cache()

    if a.preflight:
        print("\n[2/4] Preflight determinism check ...")
        if not preflight(a.arch, a.seeds[0]):
            sys.exit("Preflight FAILED - aborting.")
    else:
        print("\n[2/4] Preflight skipped (--preflight to enable)")

    print(f"\n[3/4] Launching {len(a.seeds)} workers, one per seed ...\n")
    t0 = time.time()
    procs = []
    for s in a.seeds:
        cmd = [sys.executable, os.path.abspath(__file__),
               "--arch", a.arch, "--worker", str(s)]
        procs.append(subprocess.Popen(cmd))
    codes = [p.wait() for p in procs]
    print(f"\n  workers finished with codes {codes} "
          f"in {(time.time()-t0)/60:.1f} min")

    print("\n[4/4] Merging and summarising ...")
    summarise(a.arch, a.seeds)


if __name__ == "__main__":
    main()

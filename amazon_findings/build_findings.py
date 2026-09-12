"""
Assemble amazon_findings/ from the raw experiment outputs.

Copies the primary result CSVs and derives the summary tables that until now
existed only as console output, so every number in README.md has a file
behind it. Re-runnable; overwrites what it writes.
"""
import os
import shutil
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "amazon_findings")
DATA = os.path.join(OUT, "data")
RUNS = os.path.join(OUT, "runs")

RIGHTPAD = os.path.join(ROOT, "amazon_test_top_10_results")
LEFTPAD = os.path.join(ROOT, "amazon_polarity_results")
IMDB = os.path.join(ROOT, "results", "master_2000_runs.csv")

ARCHS = ["BiLSTM+LSTM", "BiGRU+GRU", "BiLSTM+BiLSTM", "BiGRU+BiGRU"]
UNIDIR = {"BiLSTM+LSTM", "BiGRU+GRU"}
CHANCE_MAX, ESCAPE_MIN = 55.0, 80.0

os.makedirs(DATA, exist_ok=True)
os.makedirs(RUNS, exist_ok=True)


def copy(src, dst):
    if os.path.exists(src):
        shutil.copy2(src, dst)
        print(f"  copied {os.path.relpath(dst, ROOT)}")
        return True
    print(f"  MISSING {os.path.relpath(src, ROOT)}")
    return False


print("[1/5] copying primary outputs ...")
for f in ("all_runs.csv", "summary_by_pair_arch.csv", "summary_overall.csv",
          "imdb_vs_amazon_top10.csv", "run_config.json"):
    copy(os.path.join(RIGHTPAD, f), os.path.join(DATA, f))
for s in (10, 20, 30, 40, 50):
    copy(os.path.join(RIGHTPAD, f"seed_{s}.csv"),
         os.path.join(RUNS, f"rightpad_seed_{s}.csv"))
    copy(os.path.join(LEFTPAD, f"top10_seed_{s}.csv"),
         os.path.join(RUNS, f"leftpad_partial_seed_{s}.csv"))

d = pd.read_csv(os.path.join(DATA, "all_runs.csv"))
if "cell" not in d.columns:
    d["cell"] = d.act1 + "+" + d.act2

print("[2/5] escape rate by architecture ...")
rows = []
for a in ARCHS:
    g = d[d.experiment == a]
    rows.append(dict(
        architecture=a,
        second_layer="unidirectional" if a in UNIDIR else "bidirectional",
        readout="h[-1] (forward only)" if a in UNIDIR
                else "cat(h[0] fwd, h[1] bwd)",
        runs=len(g),
        escaped=int((g.accuracy >= ESCAPE_MIN).sum()),
        at_chance=int((g.accuracy <= CHANCE_MAX).sum()),
        exactly_50=int((g.accuracy == 50.0).sum()),
        mean_accuracy=round(g.accuracy.mean(), 2),
        sd_accuracy=round(g.accuracy.std(ddof=1), 2),
        min_accuracy=round(g.accuracy.min(), 2),
        max_accuracy=round(g.accuracy.max(), 2)))
esc = pd.DataFrame(rows)
esc.to_csv(os.path.join(DATA, "escape_rate_by_architecture.csv"), index=False)
print(esc.to_string(index=False))

print("\n[3/5] within-architecture rankings ...")
frames = []
for a in ARCHS:
    g = d[d.experiment == a]
    agg = g.groupby("cell").agg(
        accuracy=("accuracy", "mean"), accuracy_sd=("accuracy", "std"),
        precision=("precision", "mean"), precision_sd=("precision", "std"),
        recall=("recall", "mean"), recall_sd=("recall", "std"),
        f1=("f1", "mean"), f1_sd=("f1", "std"),
        n_seeds=("accuracy", "count")).sort_values("accuracy", ascending=False)
    agg.insert(0, "rank_within_10", range(1, len(agg) + 1))
    agg.insert(0, "architecture", a)
    agg["seeds_escaped"] = g[g.accuracy >= ESCAPE_MIN].groupby(
        "cell").size().reindex(agg.index).fillna(0).astype(int)
    agg["seeds_at_chance"] = g[g.accuracy <= CHANCE_MAX].groupby(
        "cell").size().reindex(agg.index).fillna(0).astype(int)
    frames.append(agg.reset_index())
within = pd.concat(frames, ignore_index=True).round(4)
within.to_csv(os.path.join(DATA, "ranking_within_architecture.csv"), index=False)
print(f"  wrote {len(within)} rows (10 pairs x 4 architectures)")

print("[4/5] overall ranking + IMDb comparison ...")
arch_mean = d.groupby(["cell", "experiment"]).accuracy.mean()
overall = arch_mean.groupby("cell").mean().sort_values(ascending=False)
ov = pd.DataFrame({
    "rank_within_10": range(1, len(overall) + 1),
    "amazon_accuracy": overall.round(4),
    "amazon_sd_20runs": d.groupby("cell").accuracy.std(ddof=1).reindex(
        overall.index).round(4),
    "seeds_escaped_of_20": d[d.accuracy >= ESCAPE_MIN].groupby("cell").size()
        .reindex(overall.index).fillna(0).astype(int),
})
im = pd.read_csv(IMDB)
im = im[im.cell.isin(overall.index)]
im_overall = im.groupby(["cell", "experiment"]).accuracy.mean() \
               .groupby("cell").mean()
ov["imdb_accuracy"] = im_overall.reindex(overall.index).round(4)
ov["imdb_rank_within_10"] = ov.imdb_accuracy.rank(ascending=False).astype(int)
ov["rank_shift"] = ov.imdb_rank_within_10 - ov.rank_within_10
ov["accuracy_delta"] = (ov.amazon_accuracy - ov.imdb_accuracy).round(4)
ov.to_csv(os.path.join(DATA, "ranking_overall_vs_imdb.csv"))
rho, p = spearmanr(ov.imdb_accuracy, ov.amazon_accuracy)
print(ov.to_string())
print(f"  Spearman rho = {rho:+.4f}  p = {p:.4f}")

print("[5/5] corpus length profile ...")
# Measured with the pipeline's own tokenisation; see survey_amazon_corpora.py
# (streamed sample) and the encoded caches (exact).
prof = pd.DataFrame([
    dict(corpus="IMDb (dataset/IMDB Dataset.csv)", source="encoded cache",
         median_tokens=176, pct_filling_200=42.1, median_trailing_pad=24),
    dict(corpus="Amazon Polarity (this study)", source="encoded cache",
         median_tokens=71, pct_filling_200=0.1, median_trailing_pad=129),
    dict(corpus="Amazon-2023 Movies_and_TV", source="streamed 3k sample",
         median_tokens=27, pct_filling_200=7.9, median_trailing_pad=173),
    dict(corpus="Amazon-2023 Books", source="streamed 3k sample",
         median_tokens=70, pct_filling_200=24.5, median_trailing_pad=130),
    dict(corpus="Amazon-2023 CDs_and_Vinyl", source="streamed 3k sample",
         median_tokens=68, pct_filling_200=27.0, median_trailing_pad=132),
    dict(corpus="Amazon-2023 Kindle_Store", source="streamed 3k sample",
         median_tokens=58, pct_filling_200=11.0, median_trailing_pad=142),
    dict(corpus="Amazon-2023 Video_Games", source="streamed 3k sample",
         median_tokens=44, pct_filling_200=11.9, median_trailing_pad=156),
])
prof.to_csv(os.path.join(DATA, "corpus_length_profile.csv"), index=False)
print(prof.to_string(index=False))

summary = dict(
    total_runs=int(len(d)),
    unique_combinations=int(len(d.drop_duplicates(
        subset=["cell", "experiment", "seed"]))),
    expected=200,
    complete=bool(len(d.drop_duplicates(
        subset=["cell", "experiment", "seed"])) == 200),
    runs_at_chance=int((d.accuracy <= CHANCE_MAX).sum()),
    runs_escaped=int((d.accuracy >= ESCAPE_MIN).sum()),
    runs_exactly_50=int((d.accuracy == 50.0).sum()),
    spearman_rho_vs_imdb=round(float(rho), 4),
    spearman_p_vs_imdb=round(float(p), 4),
    amazon_mean_accuracy=round(float(d.accuracy.mean()), 2),
    imdb_mean_accuracy=round(float(im.accuracy.mean()), 2),
)
json.dump(summary, open(os.path.join(DATA, "headline_numbers.json"), "w"),
          indent=2)
print(f"\nwrote {os.path.relpath(DATA, ROOT)}/headline_numbers.json")
print(json.dumps(summary, indent=2))

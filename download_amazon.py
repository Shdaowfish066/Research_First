"""
Download AmazonPolarityClassification through MTEB and cache the raw text.

MTEB is used only to resolve and fetch the dataset at its pinned revision
(mteb/amazon_polarity @ e2d317d...), so the corpus is byte-identical to what
the benchmark evaluates on. MTEB's own evaluator is NOT used: it scores frozen
sentence embeddings with a kNN/logreg probe, which is a different protocol from
this study, which trains recurrent models from scratch on raw text.

Writes cache/amazon/raw_{train,test}.parquet  (columns: text, label)
"""
import os
import mteb
import pandas as pd

OUT = os.path.join("cache", "amazon")
os.makedirs(OUT, exist_ok=True)

task = mteb.get_tasks(tasks=["AmazonPolarityClassification"])[0]
print("resolving:", task.metadata.dataset)
task.load_data()

ds = task.dataset
print("type:", type(ds))
# mteb 2.x wraps splits per language subset
if hasattr(ds, "keys") and "default" in ds:
    ds = ds["default"]
print("splits:", list(ds.keys()))

for split in ds.keys():
    d = ds[split]
    print(f"  {split}: {len(d):,} rows, cols={d.column_names}")
    df = d.to_pandas()
    tcol = "text" if "text" in df.columns else df.columns[0]
    lcol = "label" if "label" in df.columns else df.columns[-1]
    df = df[[tcol, lcol]].rename(columns={tcol: "text", lcol: "label"})
    p = os.path.join(OUT, f"raw_{split}.parquet")
    df.to_parquet(p, index=False)
    print(f"    -> {p}  label counts: {df.label.value_counts().to_dict()}")

print("\nDONE")

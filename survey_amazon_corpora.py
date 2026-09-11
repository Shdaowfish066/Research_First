"""
Which Amazon review corpus actually resembles IMDb?

The IMDb pipeline post-pads to MAX_LEN=200 and reads the final hidden state,
so it only works on corpora whose reviews are long enough to fill most of
those 200 slots. IMDb: median 176 tokens, 42.1% of reviews truncated at 200.
Amazon Polarity: median 71, 0.1% truncated -- which is why every run on it
collapsed to 49.8%.

This streams a sample from several candidate corpora and measures the same
statistic under the SAME tokenisation the pipeline uses (rs.clean_text then
.split()), so the numbers are directly comparable to the IMDb figures above.

Nothing is downloaded in full -- streaming pulls only the sampled shard.
"""
import sys, importlib.util
import numpy as np

spec = importlib.util.spec_from_file_location("rs", "pipeline.py")
rs = importlib.util.module_from_spec(spec); sys.modules["rs"] = rs
spec.loader.exec_module(rs)

from datasets import load_dataset

N = 3000          # reviews sampled per corpus
MAX_LEN = rs.MAX_LEN

HF = "hf://datasets/McAuley-Lab/Amazon-Reviews-2023/raw/review_categories"

# The McAuley repo ships a loader script, which datasets 5.x no longer runs,
# so the raw .jsonl shards are streamed directly instead.
# (label, data_files or hf path, is_json, text field(s), rating field or None)
CANDIDATES = [
    ("Amazon-2023 Movies_and_TV", f"{HF}/Movies_and_TV.jsonl", True,
     ("title", "text"), "rating"),
    ("Amazon-2023 Books", f"{HF}/Books.jsonl", True,
     ("title", "text"), "rating"),
    ("Amazon-2023 CDs_and_Vinyl", f"{HF}/CDs_and_Vinyl.jsonl", True,
     ("title", "text"), "rating"),
    ("Amazon-2023 Kindle_Store", f"{HF}/Kindle_Store.jsonl", True,
     ("title", "text"), "rating"),
    ("Amazon-2023 Video_Games", f"{HF}/Video_Games.jsonl", True,
     ("title", "text"), "rating"),
    ("Amazon Polarity (current)", "mteb/amazon_polarity", False,
     ("text",), None),
]


def measure(name, src, is_json, fields, rating_field):
    try:
        if is_json:
            ds = load_dataset("json", data_files=src, split="train",
                              streaming=True)
        else:
            ds = load_dataset(src, split="train", streaming=True)
    except Exception as e:
        print(f"{name:<28} UNAVAILABLE ({type(e).__name__}: {str(e)[:70]})")
        return None

    lens, ratings, n = [], [], 0
    try:
        for row in ds:
            txt = " ".join(str(row.get(f, "") or "") for f in fields)
            toks = rs.clean_text(txt).split()
            if not toks:
                continue
            lens.append(len(toks))
            if rating_field and row.get(rating_field) is not None:
                ratings.append(float(row[rating_field]))
            n += 1
            if n >= N:
                break
    except Exception as e:
        print(f"{name:<28} STREAM ERROR ({type(e).__name__}: {str(e)[:70]})")
        return None

    if not lens:
        print(f"{name:<28} no usable rows")
        return None

    lens = np.array(lens)
    capped = np.minimum(lens, MAX_LEN)
    pct_full = (lens >= MAX_LEN).mean() * 100
    med_pad = MAX_LEN - int(np.median(capped))

    # how balanced would a 1-2 vs 4-5 star binary split be?
    bal = ""
    if ratings:
        r = np.array(ratings)
        neg, pos = (r <= 2).sum(), (r >= 4).sum()
        bal = f"  |  neg {neg/len(r)*100:4.1f}%  pos {pos/len(r)*100:4.1f}%"

    print(f"{name:<28} median {int(np.median(lens)):>4} tok   "
          f"fills 200: {pct_full:>5.1f}%   trailing PAD {med_pad:>3}{bal}")
    return dict(name=name, median=int(np.median(lens)), pct_full=pct_full,
                med_pad=med_pad)


if __name__ == "__main__":
    print(f"sampling {N} reviews per corpus, tokenised exactly as the pipeline does\n")
    print(f"{'corpus':<28} {'length profile':<52}")
    print("-" * 100)
    print(f"{'IMDb (target)':<28} median  176 tok   fills 200:  42.1%   "
          f"trailing PAD  24  |  the profile to match")
    print("-" * 100)
    for c in CANDIDATES:
        measure(*c)
    print("-" * 100)
    print("\nA corpus is a drop-in replacement if its median is near 176 and a")
    print("large share of reviews fill all 200 slots. Then the existing")
    print("post-padding pipeline runs unchanged, with no deviation to document.")

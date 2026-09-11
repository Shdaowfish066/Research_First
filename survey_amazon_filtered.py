"""
No Amazon corpus matches IMDb's length profile as-shipped. But the long
reviews exist -- they are just diluted by one-line ones. This measures what
a length filter yields.

For each candidate: stream reviews, keep only those with >= MIN_TOK tokens
AND a non-neutral rating (1-2 = negative, 4-5 = positive, 3 dropped, exactly
how IMDb was built), then report the surviving profile and the yield rate.

Yield matters: we need 25,000 positive and 25,000 negative long reviews to
match IMDb's 50k. The scarce class is negative, so the binding constraint is
(rows streamed) x (fraction long) x (fraction negative).
"""
import sys, importlib.util
import numpy as np

spec = importlib.util.spec_from_file_location("rs", "pipeline.py")
rs = importlib.util.module_from_spec(spec); sys.modules["rs"] = rs
spec.loader.exec_module(rs)

from datasets import load_dataset

HF = "hf://datasets/McAuley-Lab/Amazon-Reviews-2023/raw/review_categories"
SCAN = 40_000       # rows streamed per corpus
MIN_TOK = 150       # keep reviews at least this long
NEED_PER_CLASS = 25_000

# category -> total review count in the 2023 dump (McAuley-Lab dataset card)
TOTALS = {
    "Movies_and_TV": 17_300_000,
    "Books": 29_500_000,
    "CDs_and_Vinyl": 4_800_000,
}


def survey(name, fname):
    ds = load_dataset("json", data_files=f"{HF}/{fname}.jsonl",
                      split="train", streaming=True)
    kept_len, kept_lab, n = [], [], 0
    for row in ds:
        n += 1
        if n > SCAN:
            break
        r = row.get("rating")
        if r is None or float(r) == 3.0:
            continue
        txt = " ".join(str(row.get(f, "") or "") for f in ("title", "text"))
        toks = rs.clean_text(txt).split()
        if len(toks) < MIN_TOK:
            continue
        kept_len.append(len(toks))
        kept_lab.append(1 if float(r) >= 4 else 0)

    if not kept_len:
        print(f"{name:<24} nothing survived the filter")
        return

    L = np.array(kept_len)
    lab = np.array(kept_lab)
    capped = np.minimum(L, rs.MAX_LEN)
    keep_rate = len(L) / SCAN
    neg_rate = (lab == 0).mean()

    # negatives per row streamed is what limits the build
    neg_per_row = keep_rate * neg_rate
    total = TOTALS.get(fname, 0)
    avail_neg = int(total * neg_per_row)

    print(f"{name:<24} median {int(np.median(capped)):>4} tok   "
          f"fills 200: {(L >= rs.MAX_LEN).mean()*100:>5.1f}%   "
          f"trailing PAD {rs.MAX_LEN - int(np.median(capped)):>3}   "
          f"keep {keep_rate*100:>4.1f}%   neg {neg_rate*100:>4.1f}%")
    verdict = "ENOUGH" if avail_neg >= NEED_PER_CLASS else "TOO FEW"
    print(f"{'':<24} -> est. {avail_neg:,} long negative reviews in the full "
          f"category ({verdict}, need {NEED_PER_CLASS:,})")


if __name__ == "__main__":
    print(f"streaming {SCAN:,} rows per corpus; keeping reviews >= {MIN_TOK} "
          f"tokens with a non-neutral rating\n")
    print(f"{'IMDb (target)':<24} median  176 tok   fills 200:  42.1%   "
          f"trailing PAD  24")
    print("-" * 104)
    for nm, fn in (("Movies_and_TV (filtered)", "Movies_and_TV"),
                   ("Books (filtered)", "Books"),
                   ("CDs_and_Vinyl (filtered)", "CDs_and_Vinyl")):
        try:
            survey(nm, fn)
        except Exception as e:
            print(f"{nm:<24} ERROR {type(e).__name__}: {str(e)[:60]}")

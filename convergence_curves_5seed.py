"""
=============================================================================
PER-EPOCH CONVERGENCE CURVES, FIVE SEEDS  (Fig. 1 data)
=============================================================================
The published Fig. 1 plotted one seed. This logs the full validation-loss
trajectory for the ten homogeneous pairs (X+X) on BiGRU+GRU under five seeds,
so the figure can show a mean curve with a +/- 1 sd band -- the error bars
Reviewer 3 asked for, applied to the convergence claim as well as the table.

PIPELINE FIDELITY: replicates convergence_rerun.py exactly -- its PaperGRU
(dropout= argument present, o[:,-1,:] readout), and ONLY torch.manual_seed and
np.random.seed, with no cudnn flags. Setting cudnn.deterministic perturbs the
numerics enough to flip epoch-1 outcomes, because epoch 1 is a knife-edge: the
model either escapes the ~50% plateau in one pass or it does not. Verified:
under these conditions seed 42 reproduces the published Table VII bit-for-bit.

Output: results/convergence_curves_5seed.csv
        columns: af, seed, epoch, val_loss, val_acc
=============================================================================
"""

import re, csv, time
from collections import Counter
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import warnings; warnings.filterwarnings("ignore")

VOCAB_SIZE, MAX_LEN, EMBED_DIM = 5000, 200, 128
GRU1, GRU2, D1, D2 = 64, 32, 64, 32
DROPOUT, BATCH, EPOCHS = 0.4, 64, 10
SEEDS = [42, 0, 1, 2, 3]
AFS = ["ReLU", "ELU", "LeakyReLU", "Tanh", "GELU", "SiLU", "Mish", "SELU",
       "PReLU", "Hardswish"]
OUT = "results/convergence_curves_5seed.csv"
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Mish(nn.Module):
    def forward(self, x): return x * torch.tanh(F.softplus(x))


ACT = {"ReLU": lambda: nn.ReLU(), "ELU": lambda: nn.ELU(alpha=1.0),
       "LeakyReLU": lambda: nn.LeakyReLU(0.01), "Tanh": lambda: nn.Tanh(),
       "GELU": lambda: nn.GELU(), "SiLU": lambda: nn.SiLU(), "Mish": lambda: Mish(),
       "SELU": lambda: nn.SELU(), "PReLU": lambda: nn.PReLU(),
       "Hardswish": lambda: nn.Hardswish()}


class PaperGRU(nn.Module):                      # verbatim convergence_rerun.py
    def __init__(s, a1, a2):
        super().__init__()
        s.embedding = nn.Embedding(VOCAB_SIZE, EMBED_DIM, padding_idx=0)
        s.bigru = nn.GRU(EMBED_DIM, GRU1, batch_first=True, bidirectional=True,
                         dropout=DROPOUT)
        s.gru = nn.GRU(GRU1 * 2, GRU2, batch_first=True)
        s.drop = nn.Dropout(DROPOUT)
        s.dense1 = nn.Linear(GRU2, D1); s.act1 = ACT[a1]()
        s.dense2 = nn.Linear(D1, D2);   s.act2 = ACT[a2]()
        s.output = nn.Linear(D2, 1)

    def forward(s, x):
        e = s.embedding(x); o, _ = s.bigru(e); o, _ = s.gru(o); h = o[:, -1, :]
        h = s.drop(h); h = s.act1(s.dense1(h)); h = s.act2(s.dense2(h))
        return s.output(h).squeeze(1)


class DS(Dataset):
    def __init__(s, X, y):
        s.X = torch.tensor(X, dtype=torch.long); s.y = torch.tensor(y, dtype=torch.float32)
    def __len__(s): return len(s.y)
    def __getitem__(s, i): return s.X[i], s.y[i]


def clean(t):
    t = re.sub(r"<[^>]+>", " ", t); t = re.sub(r"[^a-zA-Z\s]", " ", t)
    return t.lower().strip()


def load():
    df = pd.read_csv("dataset/IMDB Dataset.csv")
    df["review"] = df["review"].apply(clean)
    df["sentiment"] = (df["sentiment"] == "positive").astype(int)
    Xtr, _, ytr, _ = train_test_split(df["review"].values, df["sentiment"].values,
        test_size=0.20, random_state=42, stratify=df["sentiment"].values)
    c = Counter()
    for t in Xtr: c.update(t.split())
    w2i = {"<PAD>": 0, "<UNK>": 1}
    for w_, _ in c.most_common(VOCAB_SIZE - 2): w2i[w_] = len(w2i)
    out = np.zeros((len(Xtr), MAX_LEN), dtype=np.int64)
    for i, t in enumerate(Xtr):
        ids = [w2i.get(x, 1) for x in t.split()][:MAX_LEN]; out[i, :len(ids)] = ids
    return out, ytr


def run(af, seed, Xtr, ytr):
    torch.manual_seed(seed); np.random.seed(seed)
    m = PaperGRU(af, af).to(DEV)
    nv = int(len(Xtr) * 0.1)
    kw = dict(batch_size=BATCH, num_workers=0, pin_memory=True)
    tr = DataLoader(DS(Xtr[nv:], ytr[nv:]), shuffle=True, **kw)
    va = DataLoader(DS(Xtr[:nv], ytr[:nv]), shuffle=False, **kw)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5,
                                                     patience=2, min_lr=1e-6)
    crit = nn.BCEWithLogitsLoss()
    best, patience, hist = float("inf"), 0, []
    for ep in range(1, EPOCHS + 1):
        m.train()
        for Xb, yb in tr:
            Xb, yb = Xb.to(DEV), yb.to(DEV); opt.zero_grad()
            loss = crit(m(Xb), yb); loss.backward()
            nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        m.eval(); vl = cc = tt = 0
        with torch.no_grad():
            for Xb, yb in va:
                Xb, yb = Xb.to(DEV), yb.to(DEV); lo = m(Xb)
                vl += crit(lo, yb).item() * len(yb)
                cc += ((torch.sigmoid(lo) >= 0.5).long() == yb.long()).sum().item()
                tt += len(yb)
        vl /= tt
        hist.append((ep, round(vl, 4), round(cc / tt * 100, 2)))
        sch.step(vl)
        if vl < best: best, patience = vl, 0
        else:
            patience += 1
            if patience >= 3: break
    del m; torch.cuda.empty_cache()
    return hist


if __name__ == "__main__":
    Xtr, ytr = load()
    t0 = time.time(); n = 0; tot = len(AFS) * len(SEEDS)
    with open(OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["af", "seed", "epoch", "val_loss", "val_acc"])
        w.writeheader()
        for af in AFS:
            for s in SEEDS:
                for ep, vl, va_ in run(af, s, Xtr, ytr):
                    w.writerow(dict(af=af, seed=s, epoch=ep, val_loss=vl, val_acc=va_))
                fh.flush(); n += 1
            eta = (time.time() - t0) / n * (tot - n)
            print(f"[{n:2d}/{tot}] {af:10s} ETA {eta/60:.1f}m", flush=True)
    print(f"\nDone in {(time.time()-t0)/60:.1f} min -> {OUT}")

"""
Why do the Amazon runs collapse to 50%?

Hypothesis: both architectures read the FINAL hidden state
(PaperGRU: h[-1]; PaperLSTM: h[-1]), and encode() post-pads to MAX_LEN=200.
IMDb reviews have a median of 176 content tokens, so the final state sits
~24 steps after the text. Amazon reviews have a median of 71, so it sits
~129 steps of zero-embedding input later, by which point the state has
decayed and carries no signal.

Four conditions, same model (BiGRU+GRU, ReLU+ReLU, seed 10), 3 epochs:
  A  Amazon, post-pad 200      -- as run, expected to fail
  B  Amazon, LEFT-pad 200      -- content ends at t=199, padding leads
  C  Amazon, post-pad 100      -- shorter tail
  D  IMDb,   post-pad 200      -- control, known to work

Only the padding side / length changes. Nothing else moves.
"""
import sys, importlib.util
import numpy as np
import torch
import torch.nn as nn

spec = importlib.util.spec_from_file_location("rs", "pipeline.py")
rs = importlib.util.module_from_spec(spec); sys.modules["rs"] = rs
spec.loader.exec_module(rs)

EPOCHS = 3


def left_pad(X):
    """Move each row's nonzero prefix to the right-hand end."""
    out = np.zeros_like(X)
    for i, row in enumerate(X):
        n = int((row != 0).sum())
        if n:
            out[i, -n:] = row[:n]
    return out


def truncate(X, L):
    return X[:, :L].copy()


def quick(X_tr, y_tr, X_te, y_te, label, seed=10):
    rs.set_seed(seed)
    model = rs.PaperGRU("ReLU", "ReLU").to(rs.DEVICE)
    tr, va, te = rs.make_loaders(X_tr, y_tr, X_te, y_te)
    opt = torch.optim.Adam(model.parameters(), lr=rs.LEARNING_RATE)
    crit = nn.BCEWithLogitsLoss()
    for ep in range(1, EPOCHS + 1):
        model.train()
        for Xb, yb in tr:
            Xb, yb = Xb.to(rs.DEVICE), yb.to(rs.DEVICE)
            opt.zero_grad()
            loss = crit(model(Xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval(); cc = tt = 0; vl = 0.0
        with torch.no_grad():
            for Xb, yb in va:
                Xb, yb = Xb.to(rs.DEVICE), yb.to(rs.DEVICE)
                lo = model(Xb)
                vl += crit(lo, yb).item() * len(yb)
                cc += ((torch.sigmoid(lo) >= 0.5).long() == yb.long()).sum().item()
                tt += len(yb)
        print(f"  {label:<28} epoch {ep}  val_loss {vl/tt:.4f}  val_acc {cc/tt*100:5.2f}%",
              flush=True)
    del model; torch.cuda.empty_cache()


if __name__ == "__main__":
    aX, ay = np.load("cache/amazon_encoded/X_train.npy"), np.load("cache/amazon_encoded/y_train.npy")
    aXt, ayt = np.load("cache/amazon_encoded/X_test.npy"), np.load("cache/amazon_encoded/y_test.npy")
    iX, iy = np.load("cache/sweep_data/X_train.npy"), np.load("cache/sweep_data/y_train.npy")
    iXt, iyt = np.load("cache/sweep_data/X_test.npy"), np.load("cache/sweep_data/y_test.npy")

    print(f"Amazon content tokens: median {int(np.median((aX!=0).sum(1)))}, "
          f"trailing PAD median {200-int(np.median((aX!=0).sum(1)))}")
    print(f"IMDb   content tokens: median {int(np.median((iX!=0).sum(1)))}, "
          f"trailing PAD median {200-int(np.median((iX!=0).sum(1)))}\n")

    print("A  Amazon, post-pad 200 (as run)")
    quick(aX, ay, aXt, ayt, "amazon post-pad 200")
    print("\nB  Amazon, LEFT-pad 200")
    quick(left_pad(aX), ay, left_pad(aXt), ayt, "amazon left-pad 200")
    print("\nC  Amazon, post-pad 100")
    quick(truncate(aX, 100), ay, truncate(aXt, 100), ayt, "amazon post-pad 100")
    print("\nD  IMDb, post-pad 200 (control)")
    quick(iX, iy, iXt, iyt, "imdb post-pad 200")

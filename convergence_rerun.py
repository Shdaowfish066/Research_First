"""
=============================================================================
CONVERGENCE SPEED RERUN — Per-Activation Epoch Logging
=============================================================================
Goal : The 400-run sweep saved only final test metrics. This script reruns
       the 10 HOMOGENEOUS pairs (X+X) on the GRU architecture with full
       per-epoch logging, so convergence speed per activation function can
       be reported empirically (mentor request 3).

Why homogeneous pairs : isolates each activation's own convergence
behavior with no confound from a second function.
Why GRU : fastest sweep (~12-13 s/epoch), and the architecture where the
paper's main marginal analysis is reported.

Runtime : ~10 runs x ~7 epochs x ~13 s  ≈  15 minutes on RTX 5080.

Output : results/convergence/convergence_per_af.csv
         columns: AF, best_epoch, epochs_trained, val_loss_e1..e10,
                  val_acc_e1..e10, lr_reductions, test_acc
=============================================================================
"""

import os, re, time
from collections import Counter
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ── PAPER-EXACT CONFIG (identical to gru_activation_sweep.py) ──────────────
VOCAB_SIZE   = 5_000
MAX_LEN      = 200
EMBED_DIM    = 128
GRU1_UNITS   = 64
GRU2_UNITS   = 32
DENSE1_UNITS = 64
DENSE2_UNITS = 32
DROPOUT_RATE = 0.4
BATCH_SIZE   = 64
EPOCHS       = 10
RESULTS_DIR  = os.path.join("results", "convergence")
os.makedirs(RESULTS_DIR, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device :", DEVICE)

class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

ACTIVATIONS = {
    "ReLU"      : lambda: nn.ReLU(),
    "ELU"       : lambda: nn.ELU(alpha=1.0),
    "LeakyReLU" : lambda: nn.LeakyReLU(0.01),
    "Tanh"      : lambda: nn.Tanh(),
    "GELU"      : lambda: nn.GELU(),
    "SiLU"      : lambda: nn.SiLU(),
    "Mish"      : lambda: Mish(),
    "SELU"      : lambda: nn.SELU(),
    "PReLU"     : lambda: nn.PReLU(),
    "Hardswish" : lambda: nn.Hardswish(),
}
ACT_NAMES = list(ACTIVATIONS.keys())

# ── DATA (verbatim from gru_activation_sweep.py) ───────────────────────────
print("\n[1/3] Loading IMDb CSV ...")

def clean_text(text):
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    return text.lower().strip()

df = pd.read_csv(os.path.join("dataset", "IMDB Dataset.csv"))
df["review"]    = df["review"].apply(clean_text)
df["sentiment"] = (df["sentiment"] == "positive").astype(int)

X_train_text, X_test_text, y_train_arr, y_test_arr = train_test_split(
    df["review"].values, df["sentiment"].values,
    test_size=0.20, random_state=42, stratify=df["sentiment"].values
)

counter = Counter()
for t in X_train_text:
    counter.update(t.split())
word2idx = {"<PAD>": 0, "<UNK>": 1}
for w, _ in counter.most_common(VOCAB_SIZE - 2):
    word2idx[w] = len(word2idx)

def encode(texts):
    unk = word2idx["<UNK>"]
    out = np.zeros((len(texts), MAX_LEN), dtype=np.int64)
    for i, text in enumerate(texts):
        ids = [word2idx.get(w, unk) for w in text.split()][:MAX_LEN]
        out[i, :len(ids)] = ids
    return out

X_train = encode(X_train_text)
X_test  = encode(X_test_text)

class IMDbDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

def make_loaders(X_tr, y_tr, X_te, y_te, val_split=0.1):
    n_val = int(len(X_tr) * val_split)
    kw = dict(batch_size=BATCH_SIZE, num_workers=0, pin_memory=(DEVICE.type == "cuda"))
    return (DataLoader(IMDbDataset(X_tr[n_val:], y_tr[n_val:]), shuffle=True,  **kw),
            DataLoader(IMDbDataset(X_tr[:n_val], y_tr[:n_val]), shuffle=False, **kw),
            DataLoader(IMDbDataset(X_te, y_te),                 shuffle=False, **kw))

# ── MODEL (verbatim PaperGRU) ───────────────────────────────────────────────
class PaperGRU(nn.Module):
    def __init__(self, act1_name, act2_name):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, EMBED_DIM, padding_idx=0)
        self.bigru     = nn.GRU(EMBED_DIM, GRU1_UNITS, batch_first=True,
                                bidirectional=True, dropout=DROPOUT_RATE)
        self.gru       = nn.GRU(GRU1_UNITS * 2, GRU2_UNITS, batch_first=True)
        self.drop      = nn.Dropout(DROPOUT_RATE)
        self.dense1    = nn.Linear(GRU2_UNITS, DENSE1_UNITS)
        self.act1      = ACTIVATIONS[act1_name]()
        self.dense2    = nn.Linear(DENSE1_UNITS, DENSE2_UNITS)
        self.act2      = ACTIVATIONS[act2_name]()
        self.output    = nn.Linear(DENSE2_UNITS, 1)

    def forward(self, x):
        e    = self.embedding(x)
        o, _ = self.bigru(e)
        o, _ = self.gru(o)
        h    = o[:, -1, :]                # final hidden state
        h    = self.drop(h)
        h    = self.act1(self.dense1(h))
        h    = self.act2(self.dense2(h))
        return self.output(h).squeeze(1)

# ── TRAINING WITH FULL EPOCH LOGGING ────────────────────────────────────────
def run_logged(af_name, X_tr, y_tr, X_te, y_te):
    torch.manual_seed(42)                  # fixed seed for fair AF comparison
    np.random.seed(42)
    model = PaperGRU(af_name, af_name).to(DEVICE)
    train_loader, val_loader, test_loader = make_loaders(X_tr, y_tr, X_te, y_te)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-6)
    criterion = nn.BCEWithLogitsLoss()

    val_losses, val_accs = [], []
    best_val, best_state, best_epoch, patience = float("inf"), None, 0, 0
    lr_reductions, prev_lr = 0, 1e-3

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(Xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        vl, correct, total = 0, 0, 0
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
                logits = model(Xb)
                vl += criterion(logits, yb).item() * len(yb)
                correct += ((torch.sigmoid(logits) >= 0.5).long() == yb.long()).sum().item()
                total   += len(yb)
        vl /= total
        val_losses.append(round(vl, 4))
        val_accs.append(round(correct / total * 100, 2))
        scheduler.step(vl)

        cur_lr = optimizer.param_groups[0]["lr"]
        if cur_lr < prev_lr:
            lr_reductions += 1
            prev_lr = cur_lr

        if vl < best_val:
            best_val, best_epoch, patience = vl, epoch, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= 3:
                break

    epochs_trained = epoch
    model.load_state_dict(best_state)
    model.eval()
    preds = []
    with torch.no_grad():
        for Xb, _ in test_loader:
            logits = model(Xb.to(DEVICE))
            preds.extend((torch.sigmoid(logits) >= 0.5).long().cpu().tolist())
    test_acc = round(accuracy_score(y_te, np.array(preds)) * 100, 2)

    # pad epoch lists to length 10 for uniform CSV columns
    val_losses += [None] * (EPOCHS - len(val_losses))
    val_accs   += [None] * (EPOCHS - len(val_accs))

    del model; torch.cuda.empty_cache()
    return best_epoch, epochs_trained, lr_reductions, test_acc, val_losses, val_accs

# ── SWEEP THE 10 HOMOGENEOUS PAIRS ──────────────────────────────────────────
print("\n[2/3] Running 10 homogeneous pairs (X+X) on GRU with epoch logging ...\n")
records = []
t_total = time.time()

for i, af in enumerate(ACT_NAMES, 1):
    t0 = time.time()
    be, et, lrr, acc, vls, vas = run_logged(af, X_train, y_train_arr, X_test, y_test_arr)
    row = {"AF": af, "best_epoch": be, "epochs_trained": et,
           "lr_reductions": lrr, "test_acc": acc}
    for e in range(EPOCHS):
        row[f"val_loss_e{e+1}"] = vls[e]
        row[f"val_acc_e{e+1}"]  = vas[e]
    records.append(row)
    print(f"  [{i:2d}/10] {af:<10}  best_epoch={be}  trained={et}  "
          f"lr_cuts={lrr}  test_acc={acc}%  [{time.time()-t0:.0f}s]")

out = pd.DataFrame(records)
out.to_csv(os.path.join(RESULTS_DIR, "convergence_per_af.csv"), index=False)
print(f"\n[3/3] Saved: {RESULTS_DIR}/convergence_per_af.csv")
print(f"Total time: {(time.time()-t_total)/60:.1f} min")
print("\nSummary (sorted by best_epoch = fastest convergence first):")
print(out[["AF","best_epoch","epochs_trained","lr_reductions","test_acc"]]
      .sort_values("best_epoch").to_string(index=False))

"""
=============================================================================
SHARED TRAINING PIPELINE  (iCONEECT 2026)
=============================================================================
Single source of truth for the experiment: preprocessing, the four
architectures, and the train/eval loop. Transcribed verbatim from the
original sweep scripts (activation_sweep.py, gru_activation_sweep.py,
bilstm_bigru_sweep.py), which were verified identical to one another by diff.

Imported by run_full_sweep.py. Not runnable on its own.

-----------------------------------------------------------------------------
SEED HANDLING  --  read this before changing anything
-----------------------------------------------------------------------------
The train/test split is held FIXED at random_state=42 for every run and is
never a function of the seed. This isolates initialisation variance from
split variance, so the reported standard deviations answer the question the
reviewers actually asked.

set_seed(s) sets random.seed, numpy.random.seed, torch.manual_seed,
torch.cuda.manual_seed_all, cudnn.deterministic=True and cudnn.benchmark=
False, and is called immediately before each model and its DataLoaders are
built. The seed therefore controls weight initialisation, dropout masks and
training batch order -- not the split, the vocabulary or the encoding.

The original 400-run sweep set no seed at all: only the split carried
random_state=42, so its per-cell numbers came from an uncontrolled RNG
state. That is precisely why every cell is re-run here under five explicit
seeds.

-----------------------------------------------------------------------------
ONE DELIBERATE DEVIATION from the original sweep scripts
-----------------------------------------------------------------------------
They passed dropout=DROPOUT_RATE to the first recurrent layer, which has
num_layers=1. PyTorch ignores inter-layer dropout when num_layers==1, so the
argument is numerically inert; on torch 2.11.0+cu128 it additionally crashes
the process at exit after any cuDNN RNN forward. It is dropped here.

Verified inert before removing: under the same seed an nn.GRU built with and
without the argument has bitwise identical parameters AND bitwise identical
outputs, in train and eval mode, on both CPU and CUDA; parameter counts are
unchanged. The real nn.Dropout(DROPOUT_RATE) layer is untouched.

Input : dataset/IMDB Dataset.csv
=============================================================================
"""

import os
import re
import csv
import time
import random
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
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score
)

# ─────────────────────────────────────────────────────────────────────────────
# PAPER-EXACT CONFIG  (Table III — identical to the sweep scripts)
# ─────────────────────────────────────────────────────────────────────────────
VOCAB_SIZE   = 5_000
MAX_LEN      = 200
EMBED_DIM    = 128
RNN1_UNITS   = 64
RNN2_UNITS   = 32
BIRNN2_OUT   = RNN2_UNITS * 2     # both-bidirectional variants concat fwd+bwd
DENSE1_UNITS = 64
DENSE2_UNITS = 32
DROPOUT_RATE = 0.4
BATCH_SIZE   = 64
EPOCHS       = 10
LEARNING_RATE = 1e-3
PATIENCE_STOP = 3                 # early-stopping patience on val loss
VAL_SPLIT     = 0.1

SPLIT_STATE = 42                  # NEVER varied — see SEED HANDLING above

DATA_PATH   = os.path.join("dataset", "IMDB Dataset.csv")
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device :", DEVICE)
if DEVICE.type == "cuda":
    print("GPU    :", torch.cuda.get_device_name(0))

# Original sweep CSVs, used only for the seed-42 delta report.

# ─────────────────────────────────────────────────────────────────────────────
# ACTIVATION REGISTRY  (verbatim from the sweep scripts)
# ─────────────────────────────────────────────────────────────────────────────
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


def set_seed(seed):
    """Everything the per-run seed controls. See SEED HANDLING at the top."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ─────────────────────────────────────────────────────────────────────────────
# DATA  (paper-exact pipeline, verbatim from the sweep scripts)
# ─────────────────────────────────────────────────────────────────────────────
def clean_text(text):
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    return text.lower().strip()


def load_data():
    if not os.path.exists(DATA_PATH):
        raise SystemExit(
            f"\nERROR: {DATA_PATH} not found.\n"
            "The sweep scripts read the IMDb 50k CSV from dataset/IMDB Dataset.csv\n"
            "(the dataset/ directory is git-ignored). Place the file there and rerun."
        )

    print("\n[1/3] Loading IMDb CSV ...")
    df = pd.read_csv(DATA_PATH)
    df["review"]    = df["review"].apply(clean_text)
    df["sentiment"] = (df["sentiment"] == "positive").astype(int)

    # Split is fixed at random_state=42 for every run, by design.
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        df["review"].values, df["sentiment"].values,
        test_size=0.20, random_state=SPLIT_STATE, stratify=df["sentiment"].values
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

    X_train, X_test = encode(X_train_text), encode(X_test_text)
    print(f"  Train: {X_train.shape}  |  Test: {X_test.shape}")
    return X_train, y_train, X_test, y_test


class IMDbDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def make_loaders(X_tr, y_tr, X_te, y_te, val_split=VAL_SPLIT):
    n_val = int(len(X_tr) * val_split)
    kw = dict(batch_size=BATCH_SIZE, num_workers=0,
              pin_memory=(DEVICE.type == "cuda"))
    return (DataLoader(IMDbDataset(X_tr[n_val:], y_tr[n_val:]), shuffle=True,  **kw),
            DataLoader(IMDbDataset(X_tr[:n_val], y_tr[:n_val]), shuffle=False, **kw),
            DataLoader(IMDbDataset(X_te, y_te),                 shuffle=False, **kw))


# ─────────────────────────────────────────────────────────────────────────────
# MODELS  (from the sweep scripts)
#
# ONE deliberate deviation from the sweep scripts, and only one:
#
#   The sweep passed dropout=DROPOUT_RATE to the FIRST recurrent layer, which
#   has num_layers=1. PyTorch ignores inter-layer dropout when num_layers==1
#   (it emits a UserWarning saying so), making the argument numerically inert.
#   On torch 2.11.0+cu128 that inert argument makes the process die at exit
#   with STATUS_STACK_BUFFER_OVERRUN (0xC0000409) after any cuDNN RNN forward
#   -- results are written correctly, then the interpreter crashes on the way
#   out, which would break any chained command that follows it.
#
#   It is therefore dropped here. Verified inert before removing: with the
#   same seed, an nn.GRU built with and without the argument has bitwise
#   identical parameters and bitwise identical outputs in both train and eval
#   mode, on both CPU and CUDA. The numbers this script produces are unchanged.
#
# self.drop = nn.Dropout(DROPOUT_RATE) below is a real dropout layer and stays.
# ─────────────────────────────────────────────────────────────────────────────
class PaperLSTM(nn.Module):
    """BiLSTM + LSTM — the paper's original architecture."""
    def __init__(self, act1_name, act2_name):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, EMBED_DIM, padding_idx=0)
        self.bilstm    = nn.LSTM(EMBED_DIM, RNN1_UNITS, batch_first=True,
                                 bidirectional=True)
        self.lstm      = nn.LSTM(RNN1_UNITS * 2, RNN2_UNITS, batch_first=True)
        self.drop      = nn.Dropout(DROPOUT_RATE)
        self.dense1    = nn.Linear(RNN2_UNITS, DENSE1_UNITS)
        self.act1      = ACTIVATIONS[act1_name]()
        self.dense2    = nn.Linear(DENSE1_UNITS, DENSE2_UNITS)
        self.act2      = ACTIVATIONS[act2_name]()
        self.output    = nn.Linear(DENSE2_UNITS, 1)

    def forward(self, x):
        e = self.embedding(x)
        o, _ = self.bilstm(e)
        _, (h, _) = self.lstm(o)
        o = self.drop(h[-1])
        o = self.act1(self.dense1(o))
        o = self.act2(self.dense2(o))
        return self.output(o).squeeze(1)


class PaperGRU(nn.Module):
    """BiGRU + GRU — LSTM cells replaced by GRU cells, nothing else changed."""
    def __init__(self, act1_name, act2_name):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, EMBED_DIM, padding_idx=0)
        self.bigru     = nn.GRU(EMBED_DIM, RNN1_UNITS, batch_first=True,
                                bidirectional=True)
        self.gru       = nn.GRU(RNN1_UNITS * 2, RNN2_UNITS, batch_first=True)
        self.drop      = nn.Dropout(DROPOUT_RATE)
        self.dense1    = nn.Linear(RNN2_UNITS, DENSE1_UNITS)
        self.act1      = ACTIVATIONS[act1_name]()
        self.dense2    = nn.Linear(DENSE1_UNITS, DENSE2_UNITS)
        self.act2      = ACTIVATIONS[act2_name]()
        self.output    = nn.Linear(DENSE2_UNITS, 1)

    def forward(self, x):
        e = self.embedding(x)
        o, _ = self.bigru(e)
        _, h = self.gru(o)
        o = self.drop(h[-1])
        o = self.act1(self.dense1(o))
        o = self.act2(self.dense2(o))
        return self.output(o).squeeze(1)


class FullBiLSTM(nn.Module):
    """BiLSTM + BiLSTM — both recurrent layers bidirectional."""
    def __init__(self, act1_name, act2_name):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, EMBED_DIM, padding_idx=0)
        self.bilstm1   = nn.LSTM(EMBED_DIM, RNN1_UNITS, batch_first=True,
                                 bidirectional=True)
        self.bilstm2   = nn.LSTM(RNN1_UNITS * 2, RNN2_UNITS, batch_first=True,
                                 bidirectional=True)
        self.drop      = nn.Dropout(DROPOUT_RATE)
        self.dense1    = nn.Linear(BIRNN2_OUT, DENSE1_UNITS)
        self.act1      = ACTIVATIONS[act1_name]()
        self.dense2    = nn.Linear(DENSE1_UNITS, DENSE2_UNITS)
        self.act2      = ACTIVATIONS[act2_name]()
        self.output    = nn.Linear(DENSE2_UNITS, 1)

    def forward(self, x):
        e = self.embedding(x)
        o, _ = self.bilstm1(e)
        _, (h, _) = self.bilstm2(o)
        h_cat = torch.cat([h[0], h[1]], dim=1)
        o = self.drop(h_cat)
        o = self.act1(self.dense1(o))
        o = self.act2(self.dense2(o))
        return self.output(o).squeeze(1)


class FullBiGRU(nn.Module):
    """BiGRU + BiGRU — both recurrent layers bidirectional."""
    def __init__(self, act1_name, act2_name):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, EMBED_DIM, padding_idx=0)
        self.bigru1    = nn.GRU(EMBED_DIM, RNN1_UNITS, batch_first=True,
                                bidirectional=True)
        self.bigru2    = nn.GRU(RNN1_UNITS * 2, RNN2_UNITS, batch_first=True,
                                bidirectional=True)
        self.drop      = nn.Dropout(DROPOUT_RATE)
        self.dense1    = nn.Linear(BIRNN2_OUT, DENSE1_UNITS)
        self.act1      = ACTIVATIONS[act1_name]()
        self.dense2    = nn.Linear(DENSE1_UNITS, DENSE2_UNITS)
        self.act2      = ACTIVATIONS[act2_name]()
        self.output    = nn.Linear(DENSE2_UNITS, 1)

    def forward(self, x):
        e = self.embedding(x)
        o, _ = self.bigru1(e)
        _, h = self.bigru2(o)
        h_cat = torch.cat([h[0], h[1]], dim=1)
        o = self.drop(h_cat)
        o = self.act1(self.dense1(o))
        o = self.act2(self.dense2(o))
        return self.output(o).squeeze(1)


MODEL_FOR = {
    "BiLSTM+LSTM"  : PaperLSTM,
    "BiGRU+GRU"    : PaperGRU,
    "BiLSTM+BiLSTM": FullBiLSTM,
    "BiGRU+BiGRU"  : FullBiGRU,
}


# ─────────────────────────────────────────────────────────────────────────────
# TRAIN / EVAL  (verbatim from the sweep scripts, plus epoch bookkeeping)
# ─────────────────────────────────────────────────────────────────────────────
def run_one(model_cls, act1, act2, seed, X_tr, y_tr, X_te, y_te):
    set_seed(seed)                                  # before model + loaders

    model = model_cls(act1, act2).to(DEVICE)
    train_loader, val_loader, test_loader = make_loaders(X_tr, y_tr, X_te, y_te)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-6)
    criterion = nn.BCEWithLogitsLoss()

    best_val, best_state, patience = float("inf"), None, 0
    best_epoch, stop_epoch = 0, 0

    for epoch in range(1, EPOCHS + 1):
        stop_epoch = epoch
        model.train()
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(Xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        vl, total = 0.0, 0
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
                logits = model(Xb)
                vl    += criterion(logits, yb).item() * len(yb)
                total += len(yb)
        vl /= total
        scheduler.step(vl)

        if vl < best_val:
            best_val   = vl
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience   = 0
        else:
            patience += 1
            if patience >= PATIENCE_STOP:
                break

    model.load_state_dict(best_state)
    model.eval()
    preds = []
    with torch.no_grad():
        for Xb, _ in test_loader:
            logits = model(Xb.to(DEVICE))
            preds.extend((torch.sigmoid(logits) >= 0.5).long().cpu().tolist())
    y_pred = np.array(preds)

    metrics = dict(
        accuracy  = round(accuracy_score(y_te, y_pred) * 100, 2),
        precision = round(precision_score(y_te, y_pred, zero_division=0) * 100, 2),
        recall    = round(recall_score(y_te, y_pred, zero_division=0) * 100, 2),
        f1        = round(f1_score(y_te, y_pred, zero_division=0) * 100, 2),
        best_epoch= best_epoch,
        stop_epoch= stop_epoch,
    )
    del model
    torch.cuda.empty_cache()
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# ----------------------------------------------------------------------------
# SHARED CSV SCHEMA
# ----------------------------------------------------------------------------
FIELDS = ["act1", "act2", "experiment", "seed", "accuracy", "precision",
          "recall", "f1", "best_epoch", "stop_epoch"]

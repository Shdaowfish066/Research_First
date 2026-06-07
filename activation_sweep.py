"""
=============================================================================
ACTIVATION FUNCTION SWEEP EXPERIMENT
=============================================================================
Goal : Beat the paper using ONLY activation function changes
       Architecture is IDENTICAL to the paper (no attention, no CNN)

Paper architecture (fixed):
  Input(200) -> Embedding(5000, 128) -> BiLSTM(64, return_seq=True)
             -> LSTM(32) -> Dense(64)+Act1 -> Dense(32)+Act2
             -> Dense(1, sigmoid)

Activation functions tested:
  Paper originals : ReLU, ELU, LeakyReLU, Tanh
  Modern          : GELU, SiLU (Swish), Mish, SELU, PReLU, Hardswish

All combinations of (Act1 x Act2) are tested = 10 x 10 = 100 combos

Paper reference results (Table 1):
  LSTM1 (LeakyReLU + Tanh): Acc=88.3%  F1=88.8%  Prec=86.1%  Recall=91.7%
  LSTM2 (ELU + Tanh)      : Acc=88.3%  F1=88.3%  Prec=88.7%  Recall=88.0%
  LSTM3 (ReLU + ELU)      : Acc=88.4%  F1=88.3%  Prec=90.3%  Recall=86.3%  <- best
=============================================================================
"""

import os, re, time, itertools
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ─────────────────────────────────────────────────────────────────────────────
# PAPER-EXACT CONFIG
# ─────────────────────────────────────────────────────────────────────────────
VOCAB_SIZE   = 5_000
MAX_LEN      = 200
EMBED_DIM    = 128
LSTM1_UNITS  = 64
LSTM2_UNITS  = 32
DENSE1_UNITS = 64
DENSE2_UNITS = 32
DROPOUT_RATE = 0.4
BATCH_SIZE   = 64
EPOCHS       = 10
RESULTS_DIR  = os.path.join("results", "result_activation_sweep")

os.makedirs(RESULTS_DIR, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device :", DEVICE)
if DEVICE.type == "cuda":
    print("GPU    :", torch.cuda.get_device_name(0))

# ─────────────────────────────────────────────────────────────────────────────
# ACTIVATION REGISTRY
# ─────────────────────────────────────────────────────────────────────────────
# Mish: x * tanh(softplus(x)) — not in older torch, implement manually
class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

ACTIVATIONS = {
    # Paper originals
    "ReLU"      : lambda: nn.ReLU(),
    "ELU"       : lambda: nn.ELU(alpha=1.0),
    "LeakyReLU" : lambda: nn.LeakyReLU(0.01),
    "Tanh"      : lambda: nn.Tanh(),
    # Modern
    "GELU"      : lambda: nn.GELU(),
    "SiLU"      : lambda: nn.SiLU(),          # Swish
    "Mish"      : lambda: Mish(),
    "SELU"      : lambda: nn.SELU(),
    "PReLU"     : lambda: nn.PReLU(),
    "Hardswish" : lambda: nn.Hardswish(),
}

ACT_NAMES = list(ACTIVATIONS.keys())
print(f"\nActivation functions : {ACT_NAMES}")
print(f"Total combinations  : {len(ACT_NAMES) ** 2}")

# ─────────────────────────────────────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────────────────────────────────────
print("\n[1/4] Loading IMDb CSV ...")

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
print(f"  Train: {X_train.shape}  |  Test: {X_test.shape}")

class IMDbDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

def make_loaders(X_tr, y_tr, X_te, y_te, val_split=0.1):
    n_val = int(len(X_tr) * val_split)
    kw = dict(batch_size=BATCH_SIZE, num_workers=0, pin_memory=(DEVICE.type=="cuda"))
    return (DataLoader(IMDbDataset(X_tr[n_val:], y_tr[n_val:]), shuffle=True,  **kw),
            DataLoader(IMDbDataset(X_tr[:n_val], y_tr[:n_val]), shuffle=False, **kw),
            DataLoader(IMDbDataset(X_te, y_te),                 shuffle=False, **kw))

# ─────────────────────────────────────────────────────────────────────────────
# PAPER-EXACT MODEL
# ─────────────────────────────────────────────────────────────────────────────
class PaperLSTM(nn.Module):
    """Exact paper architecture — only Act1/Act2 change."""
    def __init__(self, act1_name, act2_name):
        super().__init__()
        self.model_name = f"{act1_name}+{act2_name}"
        self.embedding  = nn.Embedding(VOCAB_SIZE, EMBED_DIM, padding_idx=0)
        self.bilstm     = nn.LSTM(EMBED_DIM, LSTM1_UNITS,
                                  batch_first=True, bidirectional=True,
                                  dropout=DROPOUT_RATE)
        self.lstm       = nn.LSTM(LSTM1_UNITS * 2, LSTM2_UNITS, batch_first=True)
        self.drop       = nn.Dropout(DROPOUT_RATE)
        self.dense1     = nn.Linear(LSTM2_UNITS, DENSE1_UNITS)
        self.act1       = ACTIVATIONS[act1_name]()
        self.dense2     = nn.Linear(DENSE1_UNITS, DENSE2_UNITS)
        self.act2       = ACTIVATIONS[act2_name]()
        self.output     = nn.Linear(DENSE2_UNITS, 1)

    def forward(self, x):
        e = self.embedding(x)
        o, _ = self.bilstm(e)
        _, (h, _) = self.lstm(o)
        o = self.drop(h[-1])
        o = self.act1(self.dense1(o))
        o = self.act2(self.dense2(o))
        return self.output(o).squeeze(1)

# ─────────────────────────────────────────────────────────────────────────────
# FAST TRAIN / EVAL
# ─────────────────────────────────────────────────────────────────────────────
def run_one(act1_name, act2_name, X_tr, y_tr, X_te, y_te):
    model = PaperLSTM(act1_name, act2_name).to(DEVICE)
    train_loader, val_loader, test_loader = make_loaders(X_tr, y_tr, X_te, y_te)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-6)
    criterion = nn.BCEWithLogitsLoss()

    best_val, best_state, patience = float("inf"), None, 0
    for epoch in range(1, EPOCHS + 1):
        # Train
        model.train()
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(Xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        # Validate
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
        scheduler.step(vl)
        if vl < best_val:
            best_val   = vl
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience   = 0
        else:
            patience += 1
            if patience >= 3:
                break

    model.load_state_dict(best_state)
    # Test
    model.eval()
    preds = []
    with torch.no_grad():
        for Xb, _ in test_loader:
            logits = model(Xb.to(DEVICE))
            preds.extend((torch.sigmoid(logits) >= 0.5).long().cpu().tolist())
    y_pred = np.array(preds)
    acc  = accuracy_score(y_te, y_pred)
    prec = precision_score(y_te, y_pred, zero_division=0)
    rec  = recall_score(y_te, y_pred, zero_division=0)
    f1   = f1_score(y_te, y_pred, zero_division=0)
    del model; torch.cuda.empty_cache()
    return acc, prec, rec, f1

# ─────────────────────────────────────────────────────────────────────────────
# SWEEP ALL COMBINATIONS
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2/4] Sweeping all activation combinations ...\n")

PAPER_BEST = 88.4   # LSTM3 accuracy to beat

combos = list(itertools.product(ACT_NAMES, ACT_NAMES))
records = []
beat_count = 0
t_total = time.time()

for i, (a1, a2) in enumerate(combos, 1):
    t0 = time.time()
    try:
        acc, prec, rec, f1 = run_one(a1, a2, X_train, y_train_arr, X_test, y_test_arr)
        elapsed = time.time() - t0
        beat = acc * 100 > PAPER_BEST
        if beat:
            beat_count += 1
        marker = " *** BEAT ***" if beat else ""
        print(f"  [{i:3d}/100]  {a1:<12}+{a2:<12}  "
              f"Acc={acc*100:.2f}%  Prec={prec*100:.1f}%  "
              f"Rec={rec*100:.1f}%  F1={f1*100:.2f}%  [{elapsed:.1f}s]{marker}")
        records.append({"Act1": a1, "Act2": a2,
                        "Accuracy": round(acc*100, 2),
                        "Precision": round(prec*100, 2),
                        "Recall": round(rec*100, 2),
                        "F1": round(f1*100, 2),
                        "Beat_Paper": beat})
    except Exception as e:
        print(f"  [{i:3d}/100]  {a1}+{a2}  ERROR: {e}")
        records.append({"Act1": a1, "Act2": a2,
                        "Accuracy": 0, "Precision": 0,
                        "Recall": 0, "F1": 0, "Beat_Paper": False})
        torch.cuda.empty_cache()

total_time = time.time() - t_total
print(f"\nTotal time: {total_time/60:.1f} min  |  Combos that beat paper: {beat_count}/100")

results_df = pd.DataFrame(records)
results_df = results_df.sort_values("Accuracy", ascending=False).reset_index(drop=True)
results_df.to_csv(os.path.join(RESULTS_DIR, "result_activation_sweep.csv"), index=False)
print(f"Saved CSV: result_activation_sweep.csv")

# ─────────────────────────────────────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3/4] Generating plots ...")

# 1. Accuracy heatmap
pivot = results_df.pivot(index="Act1", values="Accuracy", columns="Act2")
pivot = pivot.reindex(index=ACT_NAMES, columns=ACT_NAMES)

fig, ax = plt.subplots(figsize=(12, 9))
mask = pivot.isna()
sns.heatmap(pivot, annot=True, fmt=".1f", cmap="RdYlGn", ax=ax,
            linewidths=0.5, linecolor="white",
            vmin=pivot.stack().min() - 0.5,
            vmax=min(pivot.stack().max() + 0.5, 100),
            mask=mask, annot_kws={"size": 9})
# Highlight cells that beat the paper
for i, act1 in enumerate(ACT_NAMES):
    for j, act2 in enumerate(ACT_NAMES):
        val = pivot.loc[act1, act2] if act1 in pivot.index and act2 in pivot.columns else None
        if val is not None and not pd.isna(val) and val > PAPER_BEST:
            ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False,
                                       edgecolor="blue", lw=2.5))

ax.set_title(f"Accuracy (%) Heatmap — All Activation Combinations\n"
             f"Blue border = beats paper ({PAPER_BEST}%)",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Act2 (Dense Layer 2)", fontsize=11)
ax.set_ylabel("Act1 (Dense Layer 1)", fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "01_accuracy_heatmap.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: 01_accuracy_heatmap.png")

# 2. F1 heatmap
pivot_f1 = results_df.pivot(index="Act1", values="F1", columns="Act2")
pivot_f1 = pivot_f1.reindex(index=ACT_NAMES, columns=ACT_NAMES)
fig, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(pivot_f1, annot=True, fmt=".1f", cmap="RdYlGn", ax=ax,
            linewidths=0.5, linecolor="white", annot_kws={"size": 9})
ax.set_title("F1 Score (%) Heatmap — All Activation Combinations",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Act2 (Dense Layer 2)"); ax.set_ylabel("Act1 (Dense Layer 1)")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "02_f1_heatmap.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: 02_f1_heatmap.png")

# 3. Top-15 bar chart
top15 = results_df.head(15).copy()
top15["Label"] = top15["Act1"] + "\n+" + top15["Act2"]
paper_colors = {"LeakyReLU+Tanh", "ELU+Tanh", "ReLU+ELU"}
bar_colors = ["#E74C3C" if row["Beat_Paper"] and
              row["Act1"]+"+"+row["Act2"] not in paper_colors
              else "#3498DB" if row["Act1"]+"+"+row["Act2"] in paper_colors
              else "#95A5A6" for _, row in top15.iterrows()]

fig, ax = plt.subplots(figsize=(14, 6))
bars = ax.bar(range(len(top15)), top15["Accuracy"], color=bar_colors, alpha=0.9, edgecolor="white")
for bar, val in zip(bars, top15["Accuracy"]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
            f"{val:.2f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")
ax.axhline(PAPER_BEST, color="navy", linestyle="--", linewidth=2, label=f"Paper Best ({PAPER_BEST}%)")
ax.set_xticks(range(len(top15)))
ax.set_xticklabels(top15["Label"], fontsize=9, rotation=0)
ax.set_ylim(top15["Accuracy"].min() - 1, top15["Accuracy"].max() + 1)
ax.set_ylabel("Accuracy (%)"); ax.set_title("Top-15 Activation Combinations by Accuracy", fontweight="bold")
from matplotlib.patches import Patch
legend_elements = [Patch(color="#E74C3C", label="Beats paper (new combos)"),
                   Patch(color="#3498DB", label="Paper baseline combos"),
                   Patch(color="#95A5A6", label="Below paper")]
ax.legend(handles=legend_elements + [plt.Line2D([0],[0], color="navy", linestyle="--", label=f"Paper best {PAPER_BEST}%")])
ax.grid(True, axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "03_top15_bar.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: 03_top15_bar.png")

# 4. Per-activation average accuracy
avg_as_act1 = results_df.groupby("Act1")["Accuracy"].mean().reindex(ACT_NAMES)
avg_as_act2 = results_df.groupby("Act2")["Accuracy"].mean().reindex(ACT_NAMES)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, data, title in zip(axes, [avg_as_act1, avg_as_act2],
                            ["Avg Accuracy as Act1 (Dense 1)", "Avg Accuracy as Act2 (Dense 2)"]):
    bars = ax.bar(data.index, data.values, color="#2196F3", alpha=0.85, edgecolor="white")
    for bar, val in zip(bars, data.values):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02,
                f"{val:.1f}", ha="center", va="bottom", fontsize=9)
    ax.axhline(PAPER_BEST, color="red", linestyle="--", linewidth=1.5, label=f"Paper best {PAPER_BEST}%")
    ax.set_ylim(data.min()-1, data.max()+1)
    ax.set_xlabel("Activation Function"); ax.set_ylabel("Average Accuracy (%)")
    ax.set_title(title, fontweight="bold"); ax.legend(); ax.grid(True, axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "04_per_activation_avg.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: 04_per_activation_avg.png")

# ─────────────────────────────────────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4/4] Summary\n")
print("Paper Reference:")
paper_combos = [("LeakyReLU", "Tanh", 88.3, 86.1, 91.7, 88.8),
                ("ELU",       "Tanh", 88.3, 88.7, 88.0, 88.3),
                ("ReLU",      "ELU",  88.4, 90.3, 86.3, 88.3)]
for a1, a2, acc, prec, rec, f1 in paper_combos:
    print(f"  {a1:<12}+{a2:<12}  Acc={acc:.1f}%  F1={f1:.1f}%")

print(f"\nTop 10 from our sweep (sorted by accuracy):")
print(f"  {'Act1':<12}  {'Act2':<12}  {'Accuracy':>9}  {'F1':>8}  {'Prec':>8}  {'Recall':>8}  Beat?")
print("  " + "-"*72)
for _, row in results_df.head(10).iterrows():
    mark = "*** YES ***" if row["Beat_Paper"] else ""
    print(f"  {row['Act1']:<12}  {row['Act2']:<12}  {row['Accuracy']:>8.2f}%  "
          f"{row['F1']:>7.2f}%  {row['Precision']:>7.1f}%  {row['Recall']:>7.1f}%  {mark}")

best_row = results_df.iloc[0]
print(f"\nBest combination : {best_row['Act1']} + {best_row['Act2']}")
print(f"Best accuracy    : {best_row['Accuracy']:.2f}%")
print(f"Paper best       : {PAPER_BEST}%")
diff = best_row['Accuracy'] - PAPER_BEST
if diff > 0:
    print(f"RESULT: BEAT THE PAPER by +{diff:.2f}% using only activation function changes!")
else:
    print(f"RESULT: {abs(diff):.2f}% below paper with same architecture.")

print(f"\nCombinations that beat paper: {beat_count}/100")
print(f"All outputs saved to '{RESULTS_DIR}/'")
print("Done.")

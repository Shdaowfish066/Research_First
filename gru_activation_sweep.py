"""
=============================================================================
GRU ACTIVATION FUNCTION SWEEP
=============================================================================
Goal : Replicate the LSTM activation sweep using GRU cells (apples-to-apples)
       Architecture is paper-exact — only LSTM cells replaced with GRU

Paper architecture (BiLSTM variant — now BiGRU):
  Input(200) -> Embedding(5000, 128) -> BiGRU(64, return_seq=True)
             -> GRU(32) -> Dense(64)+Act1 -> Dense(32)+Act2
             -> Dense(1, sigmoid)

Activation functions tested (same 10 as LSTM sweep):
  Paper originals : ReLU, ELU, LeakyReLU, Tanh
  Modern          : GELU, SiLU (Swish), Mish, SELU, PReLU, Hardswish

All combinations of (Act1 x Act2) = 10×10 = 100 runs
Paper reference    : LSTM3 (ReLU+ELU) Acc=88.4%, Prec=90.3%, Rec=86.3%, F1=88.3%
LSTM sweep best    : Mish+PReLU       Acc=87.66%, F1=87.51%

Outputs (results/GRU_Results/):
  result_gru_activation_sweep.csv
  01_accuracy_heatmap.png
  02_f1_heatmap.png
  03_top15_bar.png
  04_per_activation_avg.png
  05_precision_heatmap.png
  06_recall_heatmap.png
  07_confusion_matrix_top5.png
  08_gru_vs_lstm_comparison.png
=============================================================================
"""

import os, re, time, itertools
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)

# ─────────────────────────────────────────────────────────────────────────────
# PAPER-EXACT CONFIG  (identical to activation_sweep.py)
# ─────────────────────────────────────────────────────────────────────────────
VOCAB_SIZE   = 5_000
MAX_LEN      = 200
EMBED_DIM    = 128
GRU1_UNITS   = 64        # was LSTM1_UNITS
GRU2_UNITS   = 32        # was LSTM2_UNITS
DENSE1_UNITS = 64
DENSE2_UNITS = 32
DROPOUT_RATE = 0.4
BATCH_SIZE   = 64
EPOCHS       = 10
RESULTS_DIR  = os.path.join("results", "GRU_Results")
LSTM_CSV     = os.path.join("results", "result_activation_sweep",
                             "result_activation_sweep.csv")

os.makedirs(RESULTS_DIR, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device :", DEVICE)
if DEVICE.type == "cuda":
    print("GPU    :", torch.cuda.get_device_name(0))

# ─────────────────────────────────────────────────────────────────────────────
# ACTIVATION REGISTRY  (same as LSTM sweep)
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
ACT_NAMES = list(ACTIVATIONS.keys())
print(f"\nActivation functions : {ACT_NAMES}")
print(f"Total combinations  : {len(ACT_NAMES) ** 2}")

# ─────────────────────────────────────────────────────────────────────────────
# DATA  (verbatim from activation_sweep.py)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[1/5] Loading IMDb CSV ...")

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
    kw = dict(batch_size=BATCH_SIZE, num_workers=0, pin_memory=(DEVICE.type == "cuda"))
    return (DataLoader(IMDbDataset(X_tr[n_val:], y_tr[n_val:]), shuffle=True,  **kw),
            DataLoader(IMDbDataset(X_tr[:n_val], y_tr[:n_val]), shuffle=False, **kw),
            DataLoader(IMDbDataset(X_te, y_te),                 shuffle=False, **kw))

# ─────────────────────────────────────────────────────────────────────────────
# PAPER-EXACT GRU MODEL  (BiGRU + GRU replaces BiLSTM + LSTM)
# ─────────────────────────────────────────────────────────────────────────────
class PaperGRU(nn.Module):
    """Paper architecture with GRU cells — only Act1/Act2 vary."""
    def __init__(self, act1_name, act2_name):
        super().__init__()
        self.model_name = f"{act1_name}+{act2_name}"
        self.embedding  = nn.Embedding(VOCAB_SIZE, EMBED_DIM, padding_idx=0)
        # BiGRU replaces BiLSTM — GRU returns (output, hidden) not (output, (h, c))
        self.bigru      = nn.GRU(EMBED_DIM, GRU1_UNITS,
                                 batch_first=True, bidirectional=True,
                                 dropout=DROPOUT_RATE)
        # Second GRU (unidirectional) replaces second LSTM
        self.gru        = nn.GRU(GRU1_UNITS * 2, GRU2_UNITS, batch_first=True)
        self.drop       = nn.Dropout(DROPOUT_RATE)
        self.dense1     = nn.Linear(GRU2_UNITS, DENSE1_UNITS)
        self.act1       = ACTIVATIONS[act1_name]()
        self.dense2     = nn.Linear(DENSE1_UNITS, DENSE2_UNITS)
        self.act2       = ACTIVATIONS[act2_name]()
        self.output     = nn.Linear(DENSE2_UNITS, 1)

    def forward(self, x):
        e       = self.embedding(x)
        o, _    = self.bigru(e)         # _ is hidden (2, B, GRU1_UNITS) — not used
        _, h    = self.gru(o)           # h is (1, B, GRU2_UNITS)
        o       = self.drop(h[-1])      # (B, GRU2_UNITS)
        o       = self.act1(self.dense1(o))
        o       = self.act2(self.dense2(o))
        return self.output(o).squeeze(1)

# ─────────────────────────────────────────────────────────────────────────────
# TRAINING / EVALUATION
# ─────────────────────────────────────────────────────────────────────────────
def run_one(act1_name, act2_name, X_tr, y_tr, X_te, y_te, return_preds=False):
    """Train one combo and return (acc, prec, rec, f1) or with preds if requested."""
    model = PaperGRU(act1_name, act2_name).to(DEVICE)
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
        vl, total = 0.0, 0
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
                logits  = model(Xb)
                vl     += criterion(logits, yb).item() * len(yb)
                total  += len(yb)
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
    del model
    torch.cuda.empty_cache()
    if return_preds:
        return acc, prec, rec, f1, y_pred
    return acc, prec, rec, f1

# ─────────────────────────────────────────────────────────────────────────────
# SWEEP ALL 100 COMBINATIONS
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2/5] Sweeping all GRU activation combinations ...\n")

PAPER_BEST = 88.4   # LSTM3 paper accuracy

combos    = list(itertools.product(ACT_NAMES, ACT_NAMES))
records   = []
beat_count = 0
t_total   = time.time()

for i, (a1, a2) in enumerate(combos, 1):
    t0 = time.time()
    try:
        acc, prec, rec, f1 = run_one(a1, a2, X_train, y_train_arr, X_test, y_test_arr)
        elapsed = time.time() - t0
        beat    = acc * 100 > PAPER_BEST
        if beat:
            beat_count += 1
        marker = " *** BEAT ***" if beat else ""
        print(f"  [{i:3d}/100]  {a1:<12}+{a2:<12}  "
              f"Acc={acc*100:.2f}%  Prec={prec*100:.1f}%  "
              f"Rec={rec*100:.1f}%  F1={f1*100:.2f}%  [{elapsed:.1f}s]{marker}")
        records.append({"Act1": a1, "Act2": a2,
                        "Accuracy":  round(acc  * 100, 2),
                        "Precision": round(prec * 100, 2),
                        "Recall":    round(rec  * 100, 2),
                        "F1":        round(f1   * 100, 2),
                        "Beat_Paper": beat})
    except Exception as e:
        print(f"  [{i:3d}/100]  {a1}+{a2}  ERROR: {e}")
        records.append({"Act1": a1, "Act2": a2,
                        "Accuracy": 0, "Precision": 0,
                        "Recall": 0, "F1": 0, "Beat_Paper": False})
        torch.cuda.empty_cache()

total_time = time.time() - t_total
print(f"\nTotal sweep time : {total_time/60:.1f} min")
print(f"Combos beating paper (>{PAPER_BEST}%): {beat_count}/100")

results_df = pd.DataFrame(records).sort_values("Accuracy", ascending=False).reset_index(drop=True)
csv_path   = os.path.join(RESULTS_DIR, "result_gru_activation_sweep.csv")
results_df.to_csv(csv_path, index=False)
print(f"Saved CSV : {csv_path}")

# ─────────────────────────────────────────────────────────────────────────────
# RE-RUN TOP-5 COMBOS TO COLLECT PREDICTIONS (for confusion matrices)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3/5] Re-running top-5 combos for confusion matrices ...")

top5_preds = {}
for _, row in results_df.head(5).iterrows():
    a1, a2 = row["Act1"], row["Act2"]
    label  = f"{a1}+{a2}"
    print(f"  Re-training  {label} ...")
    _, _, _, _, y_pred = run_one(a1, a2, X_train, y_train_arr,
                                 X_test, y_test_arr, return_preds=True)
    top5_preds[label] = y_pred

# ─────────────────────────────────────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4/5] Generating plots ...")

def save_heatmap(metric, title, filename, paper_line=False):
    pivot = results_df.pivot(index="Act1", values=metric, columns="Act2")
    pivot = pivot.reindex(index=ACT_NAMES, columns=ACT_NAMES)
    fig, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="RdYlGn", ax=ax,
                linewidths=0.5, linecolor="white",
                vmin=pivot.stack().min() - 0.5,
                vmax=min(pivot.stack().max() + 0.5, 100),
                annot_kws={"size": 9})
    if paper_line:
        for i_, act1 in enumerate(ACT_NAMES):
            for j_, act2 in enumerate(ACT_NAMES):
                val = pivot.loc[act1, act2] if act1 in pivot.index and act2 in pivot.columns else None
                if val is not None and not pd.isna(val) and val > PAPER_BEST:
                    ax.add_patch(plt.Rectangle((j_, i_), 1, 1, fill=False,
                                               edgecolor="blue", lw=2.5))
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Act2 (Dense Layer 2)", fontsize=11)
    ax.set_ylabel("Act1 (Dense Layer 1)", fontsize=11)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {filename}")

# 1. Accuracy heatmap
save_heatmap("Accuracy",
             f"GRU — Accuracy (%) Heatmap\nBlue border = beats paper ({PAPER_BEST}%)",
             "01_accuracy_heatmap.png", paper_line=True)

# 2. F1 heatmap
save_heatmap("F1",
             "GRU — F1 Score (%) Heatmap",
             "02_f1_heatmap.png")

# 3. Top-15 bar chart
top15 = results_df.head(15).copy()
top15["Label"] = top15["Act1"] + "\n+" + top15["Act2"]
paper_combos_set = {"LeakyReLU+Tanh", "ELU+Tanh", "ReLU+ELU"}
bar_colors = []
for _, row in top15.iterrows():
    combo = f"{row['Act1']}+{row['Act2']}"
    if combo in paper_combos_set:
        bar_colors.append("#3498DB")
    elif row["Beat_Paper"]:
        bar_colors.append("#E74C3C")
    else:
        bar_colors.append("#95A5A6")

fig, ax = plt.subplots(figsize=(14, 6))
bars = ax.bar(range(len(top15)), top15["Accuracy"], color=bar_colors, alpha=0.9, edgecolor="white")
for bar, val in zip(bars, top15["Accuracy"]):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
            f"{val:.2f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")
ax.axhline(PAPER_BEST, color="navy", linestyle="--", linewidth=2,
           label=f"Paper Best LSTM3 ({PAPER_BEST}%)")
ax.set_xticks(range(len(top15)))
ax.set_xticklabels(top15["Label"], fontsize=9)
ax.set_ylim(top15["Accuracy"].min() - 1, top15["Accuracy"].max() + 1.5)
ax.set_ylabel("Accuracy (%)")
ax.set_title("GRU — Top-15 Activation Combinations by Accuracy", fontweight="bold")
legend_elements = [
    mpatches.Patch(color="#E74C3C", label="Beats paper (new GRU combos)"),
    mpatches.Patch(color="#3498DB", label="Paper baseline combos (GRU)"),
    mpatches.Patch(color="#95A5A6", label="Below paper"),
    plt.Line2D([0], [0], color="navy", linestyle="--", label=f"Paper best {PAPER_BEST}%"),
]
ax.legend(handles=legend_elements)
ax.grid(True, axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "03_top15_bar.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: 03_top15_bar.png")

# 4. Per-activation average accuracy
avg_as_act1 = results_df.groupby("Act1")["Accuracy"].mean().reindex(ACT_NAMES)
avg_as_act2 = results_df.groupby("Act2")["Accuracy"].mean().reindex(ACT_NAMES)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, data, title in zip(axes,
                            [avg_as_act1, avg_as_act2],
                            ["Avg Accuracy as Act1 (Dense 1)",
                             "Avg Accuracy as Act2 (Dense 2)"]):
    bars_ax = ax.bar(data.index, data.values, color="#2196F3", alpha=0.85, edgecolor="white")
    for bar, val in zip(bars_ax, data.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.1f}", ha="center", va="bottom", fontsize=9)
    ax.axhline(PAPER_BEST, color="red", linestyle="--", linewidth=1.5,
               label=f"Paper best {PAPER_BEST}%")
    ax.set_ylim(data.min() - 1, data.max() + 1)
    ax.set_xlabel("Activation Function")
    ax.set_ylabel("Average Accuracy (%)")
    ax.set_title(f"GRU — {title}", fontweight="bold")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "04_per_activation_avg.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: 04_per_activation_avg.png")

# 5. Precision heatmap
save_heatmap("Precision",
             "GRU — Precision (%) Heatmap",
             "05_precision_heatmap.png")

# 6. Recall heatmap
save_heatmap("Recall",
             "GRU — Recall (%) Heatmap",
             "06_recall_heatmap.png")

# 7. Confusion matrices for top-5
fig, axes = plt.subplots(1, 5, figsize=(22, 4))
for ax, (label, y_pred) in zip(axes, top5_preds.items()):
    cm = confusion_matrix(y_test_arr, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Neg", "Pos"], yticklabels=["Neg", "Pos"],
                cbar=False, linewidths=0.5)
    acc_val = accuracy_score(y_test_arr, y_pred) * 100
    ax.set_title(f"{label}\nAcc={acc_val:.2f}%", fontsize=9, fontweight="bold")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
fig.suptitle("GRU — Confusion Matrices: Top-5 Activation Combinations",
             fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "07_confusion_matrix_top5.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: 07_confusion_matrix_top5.png")

# 8. GRU vs LSTM comparison (top-10 from each)
if os.path.exists(LSTM_CSV):
    lstm_df = pd.read_csv(LSTM_CSV).sort_values("Accuracy", ascending=False).head(10)
    gru_top10 = results_df.head(10).copy()

    lstm_df["Label"]    = lstm_df["Act1"] + "+" + lstm_df["Act2"]
    gru_top10["Label"]  = gru_top10["Act1"] + "+" + gru_top10["Act2"]

    fig, axes = plt.subplots(1, 2, figsize=(18, 6), sharey=False)
    x_lstm = range(len(lstm_df))
    x_gru  = range(len(gru_top10))

    # LSTM bars
    b1 = axes[0].bar(x_lstm, lstm_df["Accuracy"], color="#3498DB", alpha=0.85, edgecolor="white")
    for bar, val in zip(b1, lstm_df["Accuracy"]):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.03,
                     f"{val:.2f}%", ha="center", va="bottom", fontsize=8)
    axes[0].axhline(PAPER_BEST, color="red", linestyle="--", linewidth=1.5,
                    label=f"Paper best {PAPER_BEST}%")
    axes[0].set_xticks(list(x_lstm))
    axes[0].set_xticklabels(lstm_df["Label"].tolist(), rotation=30, ha="right", fontsize=8)
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].set_title("LSTM Sweep — Top 10", fontweight="bold", fontsize=12)
    axes[0].legend(); axes[0].grid(True, axis="y", alpha=0.3)
    ymin = min(lstm_df["Accuracy"].min(), gru_top10["Accuracy"].min()) - 1
    ymax = max(lstm_df["Accuracy"].max(), gru_top10["Accuracy"].max()) + 1.5
    axes[0].set_ylim(ymin, ymax)

    # GRU bars
    b2 = axes[1].bar(x_gru, gru_top10["Accuracy"], color="#E74C3C", alpha=0.85, edgecolor="white")
    for bar, val in zip(b2, gru_top10["Accuracy"]):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.03,
                     f"{val:.2f}%", ha="center", va="bottom", fontsize=8)
    axes[1].axhline(PAPER_BEST, color="red", linestyle="--", linewidth=1.5,
                    label=f"Paper best {PAPER_BEST}%")
    axes[1].set_xticks(list(x_gru))
    axes[1].set_xticklabels(gru_top10["Label"].tolist(), rotation=30, ha="right", fontsize=8)
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_title("GRU Sweep — Top 10", fontweight="bold", fontsize=12)
    axes[1].legend(); axes[1].grid(True, axis="y", alpha=0.3)
    axes[1].set_ylim(ymin, ymax)

    fig.suptitle("LSTM vs GRU Activation Sweep — Top-10 Accuracy Comparison",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "08_gru_vs_lstm_comparison.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: 08_gru_vs_lstm_comparison.png")
else:
    print(f"  Skipping plot 08 — LSTM CSV not found at {LSTM_CSV}")

# ─────────────────────────────────────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print("\n[5/5] Summary\n")
print("Paper Reference (LSTM):")
paper_combos = [("LeakyReLU", "Tanh", 88.3, 86.1, 91.7, 88.8),
                ("ELU",       "Tanh", 88.3, 88.7, 88.0, 88.3),
                ("ReLU",      "ELU",  88.4, 90.3, 86.3, 88.3)]
for a1, a2, acc, prec, rec, f1 in paper_combos:
    print(f"  {a1:<12}+{a2:<12}  Acc={acc:.1f}%  Prec={prec:.1f}%  Rec={rec:.1f}%  F1={f1:.1f}%")

print(f"\nTop 10 GRU results (sorted by accuracy):")
print(f"  {'Act1':<12}  {'Act2':<12}  {'Accuracy':>9}  {'Precision':>10}  {'Recall':>8}  {'F1':>8}  Beat?")
print("  " + "-"*78)
for _, row in results_df.head(10).iterrows():
    mark = "YES" if row["Beat_Paper"] else ""
    print(f"  {row['Act1']:<12}  {row['Act2']:<12}  {row['Accuracy']:>8.2f}%  "
          f"{row['Precision']:>9.1f}%  {row['Recall']:>7.1f}%  {row['F1']:>7.2f}%  {mark}")

best = results_df.iloc[0]
print(f"\nBest GRU combination : {best['Act1']} + {best['Act2']}")
print(f"Best GRU accuracy    : {best['Accuracy']:.2f}%")
print(f"Paper best (LSTM3)   : {PAPER_BEST}%")
diff = best["Accuracy"] - PAPER_BEST
if diff > 0:
    print(f"RESULT: GRU BEATS paper by +{diff:.2f}%!")
else:
    print(f"RESULT: GRU is {abs(diff):.2f}% below paper's LSTM3.")

print(f"\nGRU combos that beat paper : {beat_count}/100")
print(f"All outputs saved to       : {RESULTS_DIR}/")
print("Done.")

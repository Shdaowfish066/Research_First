"""
=============================================================================
FULLY BIDIRECTIONAL SWEEP  —  BiLSTM + BiGRU
=============================================================================
Extension of the paper's architecture: BOTH RNN layers are now bidirectional.

Paper (original):
  Embedding → BiRNN(64) → RNN(32) → Dense(64)+Act1 → Dense(32)+Act2 → Out

This experiment:
  Embedding → BiRNN(64) → BiRNN(32) → Dense(64)+Act1 → Dense(32)+Act2 → Out

Key difference:
  Second RNN layer is now bidirectional → hidden = cat(h_fwd, h_bwd) = 64 dim
  Dense1 input: 64 (was 32)

Two models swept:
  1. FullBiLSTM  — both layers BiLSTM
  2. FullBiGRU   — both layers BiGRU

All other hyperparams: PAPER-EXACT
  VOCAB=5000, MAXLEN=200, EMBED=128, BiRNN1=64, BiRNN2=32,
  DENSE1=64, DENSE2=32, DROPOUT=0.4, BATCH=64, EPOCHS=10,
  Adam(lr=1e-3), ReduceLROnPlateau(factor=0.5,patience=2,min_lr=1e-6),
  BCEWithLogitsLoss, GradClip=1.0, EarlyStop patience=3

100 activation combos per model  →  200 total runs

Outputs:
  results/BiLSTM_Results/  —  CSV + 8 plots for FullBiLSTM
  results/BiGRU_Results/   —  CSV + 8 plots for FullBiGRU
  results/BiLSTM_Results/09_bilstm_vs_bigru_comparison.png  (final comparison)
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
# PAPER-EXACT CONFIG
# ─────────────────────────────────────────────────────────────────────────────
VOCAB_SIZE   = 5_000
MAX_LEN      = 200
EMBED_DIM    = 128
RNN1_UNITS   = 64
RNN2_UNITS   = 32
DENSE1_UNITS = 64
DENSE2_UNITS = 32
DROPOUT_RATE = 0.4
BATCH_SIZE   = 64
EPOCHS       = 10

# Dense1 receives concatenated fwd+bwd hidden from BiRNN2 → 32*2 = 64
BIRNN2_OUT   = RNN2_UNITS * 2

BILSTM_DIR   = os.path.join("results", "BiLSTM_Results")
BIGRU_DIR    = os.path.join("results", "BiGRU_Results")
PAPER_BEST   = 88.4   # LSTM3 paper accuracy

os.makedirs(BILSTM_DIR, exist_ok=True)
os.makedirs(BIGRU_DIR,  exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device :", DEVICE)
if DEVICE.type == "cuda":
    print("GPU    :", torch.cuda.get_device_name(0))

# ─────────────────────────────────────────────────────────────────────────────
# ACTIVATION REGISTRY
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
print(f"Total combinations  : {len(ACT_NAMES) ** 2}  ×  2 models  =  {len(ACT_NAMES)**2 * 2} runs")

# ─────────────────────────────────────────────────────────────────────────────
# DATA  (paper-exact pipeline)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[1/6] Loading IMDb CSV ...")

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
# MODEL DEFINITIONS
# ─────────────────────────────────────────────────────────────────────────────

class FullBiLSTM(nn.Module):
    """Both LSTM layers are bidirectional. Dense1 input = RNN2_UNITS*2 = 64."""
    def __init__(self, act1_name, act2_name):
        super().__init__()
        self.model_name = f"{act1_name}+{act2_name}"
        self.embedding  = nn.Embedding(VOCAB_SIZE, EMBED_DIM, padding_idx=0)
        self.bilstm1    = nn.LSTM(EMBED_DIM, RNN1_UNITS,
                                  batch_first=True, bidirectional=True,
                                  dropout=DROPOUT_RATE)
        # Second layer is now also bidirectional
        self.bilstm2    = nn.LSTM(RNN1_UNITS * 2, RNN2_UNITS,
                                  batch_first=True, bidirectional=True)
        self.drop       = nn.Dropout(DROPOUT_RATE)
        # Dense1 input = 32*2 = 64 (concat fwd + bwd hidden from bilstm2)
        self.dense1     = nn.Linear(BIRNN2_OUT, DENSE1_UNITS)
        self.act1       = ACTIVATIONS[act1_name]()
        self.dense2     = nn.Linear(DENSE1_UNITS, DENSE2_UNITS)
        self.act2       = ACTIVATIONS[act2_name]()
        self.output     = nn.Linear(DENSE2_UNITS, 1)

    def forward(self, x):
        e        = self.embedding(x)
        o, _     = self.bilstm1(e)                    # (B, T, 128)
        _, (h,_) = self.bilstm2(o)                    # h: (2, B, 32)
        # Concatenate forward and backward hidden states
        h_cat    = torch.cat([h[0], h[1]], dim=1)     # (B, 64)
        o        = self.drop(h_cat)
        o        = self.act1(self.dense1(o))
        o        = self.act2(self.dense2(o))
        return self.output(o).squeeze(1)


class FullBiGRU(nn.Module):
    """Both GRU layers are bidirectional. Dense1 input = RNN2_UNITS*2 = 64."""
    def __init__(self, act1_name, act2_name):
        super().__init__()
        self.model_name = f"{act1_name}+{act2_name}"
        self.embedding  = nn.Embedding(VOCAB_SIZE, EMBED_DIM, padding_idx=0)
        self.bigru1     = nn.GRU(EMBED_DIM, RNN1_UNITS,
                                 batch_first=True, bidirectional=True,
                                 dropout=DROPOUT_RATE)
        # Second layer is now also bidirectional
        self.bigru2     = nn.GRU(RNN1_UNITS * 2, RNN2_UNITS,
                                 batch_first=True, bidirectional=True)
        self.drop       = nn.Dropout(DROPOUT_RATE)
        # Dense1 input = 32*2 = 64 (concat fwd + bwd hidden from bigru2)
        self.dense1     = nn.Linear(BIRNN2_OUT, DENSE1_UNITS)
        self.act1       = ACTIVATIONS[act1_name]()
        self.dense2     = nn.Linear(DENSE1_UNITS, DENSE2_UNITS)
        self.act2       = ACTIVATIONS[act2_name]()
        self.output     = nn.Linear(DENSE2_UNITS, 1)

    def forward(self, x):
        e     = self.embedding(x)
        o, _  = self.bigru1(e)                        # (B, T, 128)
        _, h  = self.bigru2(o)                        # h: (2, B, 32)
        # Concatenate forward and backward hidden states
        h_cat = torch.cat([h[0], h[1]], dim=1)        # (B, 64)
        o     = self.drop(h_cat)
        o     = self.act1(self.dense1(o))
        o     = self.act2(self.dense2(o))
        return self.output(o).squeeze(1)

# ─────────────────────────────────────────────────────────────────────────────
# TRAIN / EVAL
# ─────────────────────────────────────────────────────────────────────────────
def run_one(model_cls, act1_name, act2_name, X_tr, y_tr, X_te, y_te,
            return_preds=False):
    model = model_cls(act1_name, act2_name).to(DEVICE)
    train_loader, val_loader, test_loader = make_loaders(X_tr, y_tr, X_te, y_te)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-6)
    criterion = nn.BCEWithLogitsLoss()

    best_val, best_state, patience = float("inf"), None, 0
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
# PLOTTING HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def make_plots(results_df, top5_preds, results_dir, model_label,
               lstm_csv_for_compare=None, gru_csv_for_compare=None):
    """Generate all 8 plots (+ optional comparison) for a given sweep."""

    def save_heatmap(metric, title, filename, mark_beat=False):
        pivot = results_df.pivot(index="Act1", values=metric, columns="Act2")
        pivot = pivot.reindex(index=ACT_NAMES, columns=ACT_NAMES)
        fig, ax = plt.subplots(figsize=(12, 9))
        sns.heatmap(pivot, annot=True, fmt=".1f", cmap="RdYlGn", ax=ax,
                    linewidths=0.5, linecolor="white",
                    vmin=pivot.stack().min() - 0.5,
                    vmax=min(pivot.stack().max() + 0.5, 100),
                    annot_kws={"size": 9})
        if mark_beat:
            for i_, a1 in enumerate(ACT_NAMES):
                for j_, a2 in enumerate(ACT_NAMES):
                    val = pivot.loc[a1, a2] if a1 in pivot.index and a2 in pivot.columns else None
                    if val is not None and not pd.isna(val) and val > PAPER_BEST:
                        ax.add_patch(plt.Rectangle((j_, i_), 1, 1, fill=False,
                                                   edgecolor="blue", lw=2.5))
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("Act2 (Dense Layer 2)", fontsize=11)
        ax.set_ylabel("Act1 (Dense Layer 1)", fontsize=11)
        plt.tight_layout()
        path = os.path.join(results_dir, filename)
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {filename}")

    # 1. Accuracy heatmap
    save_heatmap("Accuracy",
                 f"{model_label} — Accuracy (%) Heatmap\n"
                 f"Blue border = beats paper LSTM3 ({PAPER_BEST}%)",
                 "01_accuracy_heatmap.png", mark_beat=True)

    # 2. F1 heatmap
    save_heatmap("F1", f"{model_label} — F1 Score (%) Heatmap",
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
    bars = ax.bar(range(len(top15)), top15["Accuracy"],
                  color=bar_colors, alpha=0.9, edgecolor="white")
    for bar, val in zip(bars, top15["Accuracy"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                f"{val:.2f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax.axhline(PAPER_BEST, color="navy", linestyle="--", linewidth=2,
               label=f"Paper Best LSTM3 ({PAPER_BEST}%)")
    ax.set_xticks(range(len(top15)))
    ax.set_xticklabels(top15["Label"], fontsize=9)
    ax.set_ylim(top15["Accuracy"].min() - 1, top15["Accuracy"].max() + 1.5)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"{model_label} — Top-15 Activation Combinations", fontweight="bold")
    legend_elements = [
        mpatches.Patch(color="#E74C3C", label="Beats paper"),
        mpatches.Patch(color="#3498DB", label="Paper baseline combos"),
        mpatches.Patch(color="#95A5A6", label="Below paper"),
        plt.Line2D([0], [0], color="navy", linestyle="--", label=f"Paper best {PAPER_BEST}%"),
    ]
    ax.legend(handles=legend_elements)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "03_top15_bar.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: 03_top15_bar.png")

    # 4. Per-activation average accuracy
    avg_act1 = results_df.groupby("Act1")["Accuracy"].mean().reindex(ACT_NAMES)
    avg_act2 = results_df.groupby("Act2")["Accuracy"].mean().reindex(ACT_NAMES)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, data, title in zip(axes, [avg_act1, avg_act2],
                                ["Avg Accuracy as Act1 (Dense 1)",
                                 "Avg Accuracy as Act2 (Dense 2)"]):
        b = ax.bar(data.index, data.values, color="#2196F3", alpha=0.85, edgecolor="white")
        for bar, val in zip(b, data.values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{val:.1f}", ha="center", va="bottom", fontsize=9)
        ax.axhline(PAPER_BEST, color="red", linestyle="--", linewidth=1.5,
                   label=f"Paper best {PAPER_BEST}%")
        ax.set_ylim(data.min() - 1, data.max() + 1)
        ax.set_xlabel("Activation Function")
        ax.set_ylabel("Average Accuracy (%)")
        ax.set_title(f"{model_label} — {title}", fontweight="bold")
        ax.legend(); ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "04_per_activation_avg.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: 04_per_activation_avg.png")

    # 5. Precision heatmap
    save_heatmap("Precision", f"{model_label} — Precision (%) Heatmap",
                 "05_precision_heatmap.png")

    # 6. Recall heatmap
    save_heatmap("Recall", f"{model_label} — Recall (%) Heatmap",
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
    fig.suptitle(f"{model_label} — Confusion Matrices: Top-5 Combinations",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "07_confusion_matrix_top5.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: 07_confusion_matrix_top5.png")

    # 8. Compare with previous experiments (LSTM sweep + GRU sweep)
    ref_csvs = []
    if lstm_csv_for_compare and os.path.exists(lstm_csv_for_compare):
        ref_df = pd.read_csv(lstm_csv_for_compare).sort_values("Accuracy", ascending=False).head(10)
        ref_df["Label"] = ref_df["Act1"] + "+" + ref_df["Act2"]
        ref_csvs.append(("LSTM Sweep", ref_df, "#3498DB"))
    if gru_csv_for_compare and os.path.exists(gru_csv_for_compare):
        gru_df = pd.read_csv(gru_csv_for_compare).sort_values("Accuracy", ascending=False).head(10)
        gru_df["Label"] = gru_df["Act1"] + "+" + gru_df["Act2"]
        ref_csvs.append(("GRU Sweep", gru_df, "#E67E22"))

    cur_top10 = results_df.head(10).copy()
    cur_top10["Label"] = cur_top10["Act1"] + "+" + cur_top10["Act2"]
    all_accs = list(cur_top10["Accuracy"])
    for _, rdf, _ in ref_csvs:
        all_accs.extend(rdf["Accuracy"].tolist())
    y_min = min(all_accs) - 1
    y_max = max(all_accs) + 1.5

    n_panels = 1 + len(ref_csvs)
    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 6), sharey=False)
    if n_panels == 1:
        axes = [axes]

    def draw_bar(ax, data, title, color):
        x = range(len(data))
        b = ax.bar(x, data["Accuracy"], color=color, alpha=0.85, edgecolor="white")
        for bar, val in zip(b, data["Accuracy"]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.03,
                    f"{val:.2f}%", ha="center", va="bottom", fontsize=8)
        ax.axhline(PAPER_BEST, color="red", linestyle="--", linewidth=1.5,
                   label=f"Paper best {PAPER_BEST}%")
        ax.set_xticks(list(x))
        ax.set_xticklabels(data["Label"].tolist(), rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("Accuracy (%)")
        ax.set_title(title, fontweight="bold", fontsize=11)
        ax.set_ylim(y_min, y_max)
        ax.legend(); ax.grid(True, axis="y", alpha=0.3)

    for (ref_label, rdf, col), ax in zip(ref_csvs, axes):
        draw_bar(ax, rdf, f"{ref_label} — Top 10", col)
    draw_bar(axes[-1], cur_top10, f"{model_label} — Top 10", "#27AE60")

    fig.suptitle(f"Activation Sweep Comparison — Top-10 Accuracy",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "08_comparison_chart.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: 08_comparison_chart.png")


# ─────────────────────────────────────────────────────────────────────────────
# GENERIC SWEEP RUNNER
# ─────────────────────────────────────────────────────────────────────────────
def run_sweep(model_cls, model_label, results_dir,
              lstm_csv=None, gru_csv=None):
    print(f"\n{'='*70}")
    print(f"  SWEEPING: {model_label}")
    print(f"{'='*70}\n")
    combos    = list(itertools.product(ACT_NAMES, ACT_NAMES))
    records   = []
    beat_count = 0
    t_total   = time.time()

    for i, (a1, a2) in enumerate(combos, 1):
        t0 = time.time()
        try:
            acc, prec, rec, f1 = run_one(model_cls, a1, a2,
                                          X_train, y_train_arr, X_test, y_test_arr)
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

    results_df = (pd.DataFrame(records)
                  .sort_values("Accuracy", ascending=False)
                  .reset_index(drop=True))
    csv_name   = f"result_{model_label.lower().replace(' ', '_')}_sweep.csv"
    csv_path   = os.path.join(results_dir, csv_name)
    results_df.to_csv(csv_path, index=False)
    print(f"Saved CSV : {csv_path}")

    # Re-run top-5 for confusion matrices
    print(f"\nRe-running top-5 combos for confusion matrices ...")
    top5_preds = {}
    for _, row in results_df.head(5).iterrows():
        a1, a2 = row["Act1"], row["Act2"]
        label  = f"{a1}+{a2}"
        print(f"  Re-training  {label} ...")
        _, _, _, _, y_pred = run_one(model_cls, a1, a2,
                                     X_train, y_train_arr, X_test, y_test_arr,
                                     return_preds=True)
        top5_preds[label] = y_pred

    print(f"\nGenerating plots for {model_label} ...")
    make_plots(results_df, top5_preds, results_dir, model_label,
               lstm_csv_for_compare=lstm_csv,
               gru_csv_for_compare=gru_csv)

    # Summary print
    print(f"\nTop 10 {model_label} results:")
    print(f"  {'Act1':<12}  {'Act2':<12}  {'Accuracy':>9}  {'Precision':>10}  {'Recall':>8}  {'F1':>8}  Beat?")
    print("  " + "-"*78)
    for _, row in results_df.head(10).iterrows():
        mark = "YES" if row["Beat_Paper"] else ""
        print(f"  {row['Act1']:<12}  {row['Act2']:<12}  {row['Accuracy']:>8.2f}%  "
              f"{row['Precision']:>9.1f}%  {row['Recall']:>7.1f}%  {row['F1']:>7.2f}%  {mark}")

    best = results_df.iloc[0]
    diff = best["Accuracy"] - PAPER_BEST
    print(f"\nBest {model_label} : {best['Act1']} + {best['Act2']}")
    print(f"  Accuracy  : {best['Accuracy']:.2f}%")
    print(f"  Precision : {best['Precision']:.1f}%")
    print(f"  Recall    : {best['Recall']:.1f}%")
    print(f"  F1        : {best['F1']:.2f}%")
    print(f"  vs Paper  : {'+' if diff >= 0 else ''}{diff:.2f}%")
    print(f"  Combos beating paper : {beat_count}/100")
    print(f"  Outputs saved to     : {results_dir}/")

    return results_df

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
LSTM_SWEEP_CSV = os.path.join("results", "result_activation_sweep",
                               "result_activation_sweep.csv")
GRU_SWEEP_CSV  = os.path.join("results", "GRU_Results",
                               "result_gru_activation_sweep.csv")

print("\n[2/6] Running FullBiLSTM sweep ...")
bilstm_df = run_sweep(FullBiLSTM, "FullBiLSTM", BILSTM_DIR,
                      lstm_csv=LSTM_SWEEP_CSV, gru_csv=GRU_SWEEP_CSV)

print("\n[3/6] Running FullBiGRU sweep ...")
bigru_df = run_sweep(FullBiGRU, "FullBiGRU", BIGRU_DIR,
                     lstm_csv=LSTM_SWEEP_CSV, gru_csv=GRU_SWEEP_CSV)

# ─────────────────────────────────────────────────────────────────────────────
# FINAL CROSS-MODEL COMPARISON  (BiLSTM vs BiGRU top-10 side by side)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4/6] Generating BiLSTM vs BiGRU comparison chart ...")

bilstm_top10 = bilstm_df.head(10).copy()
bigru_top10  = bigru_df.head(10).copy()
bilstm_top10["Label"] = bilstm_top10["Act1"] + "+" + bilstm_top10["Act2"]
bigru_top10["Label"]  = bigru_top10["Act1"]  + "+" + bigru_top10["Act2"]

all_accs = list(bilstm_top10["Accuracy"]) + list(bigru_top10["Accuracy"])
y_min = min(all_accs) - 1
y_max = max(all_accs) + 1.5

fig, axes = plt.subplots(1, 2, figsize=(18, 6), sharey=False)
for ax, df, title, color in zip(
    axes,
    [bilstm_top10, bigru_top10],
    ["FullBiLSTM — Top 10", "FullBiGRU — Top 10"],
    ["#8E44AD", "#16A085"]
):
    x = range(len(df))
    b = ax.bar(x, df["Accuracy"], color=color, alpha=0.85, edgecolor="white")
    for bar, val in zip(b, df["Accuracy"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.03,
                f"{val:.2f}%", ha="center", va="bottom", fontsize=8)
    ax.axhline(PAPER_BEST, color="red", linestyle="--", linewidth=1.5,
               label=f"Paper best {PAPER_BEST}%")
    ax.set_xticks(list(x))
    ax.set_xticklabels(df["Label"].tolist(), rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(title, fontweight="bold", fontsize=12)
    ax.set_ylim(y_min, y_max)
    ax.legend(); ax.grid(True, axis="y", alpha=0.3)

fig.suptitle("FullBiLSTM vs FullBiGRU — Top-10 Accuracy Comparison",
             fontsize=14, fontweight="bold")
plt.tight_layout()
out_path = os.path.join(BILSTM_DIR, "09_bilstm_vs_bigru_comparison.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {out_path}")

# ─────────────────────────────────────────────────────────────────────────────
# FINAL OVERALL SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print("\n[5/6] Overall Summary\n")
print("=" * 70)
print(f"{'Model':<22} {'Best Combo':<28} {'Acc':>7}  {'Prec':>6}  {'Rec':>6}  {'F1':>6}")
print("=" * 70)

paper_rows = [
    ("Paper LSTM1", "LeakyReLU+Tanh",    88.3, 86.1, 91.7, 88.8),
    ("Paper LSTM3", "ReLU+ELU (best)",   88.4, 90.3, 86.3, 88.3),
]
for label, combo, acc, prec, rec, f1 in paper_rows:
    print(f"  {label:<20}  {combo:<26}  {acc:>6.2f}%  {prec:>5.1f}%  {rec:>5.1f}%  {f1:>5.2f}%")

for label, df in [("FullBiLSTM", bilstm_df), ("FullBiGRU", bigru_df)]:
    best = df.iloc[0]
    combo = f"{best['Act1']}+{best['Act2']}"
    print(f"  {label:<20}  {combo:<26}  {best['Accuracy']:>6.2f}%  "
          f"{best['Precision']:>5.1f}%  {best['Recall']:>5.1f}%  {best['F1']:>5.2f}%")

print("=" * 70)
print(f"\nAll outputs:")
print(f"  BiLSTM  → results/BiLSTM_Results/  (8 plots + comparison + CSV)")
print(f"  BiGRU   → results/BiGRU_Results/   (8 plots + CSV)")
print("\n[6/6] Done.")

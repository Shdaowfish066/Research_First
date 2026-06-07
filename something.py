"""
=============================================================================
Comparative Analysis of Activation Functions in LSTM Models
for Sentiment Classification on IMDb Dataset
=============================================================================
Paper: IJARCCE Vol. 14, Issue 4, April 2025
DOI: 10.17148/IJARCCE.2025.14421

Three Bi-LSTM Models (same base architecture, different dense activations):
  LSTM1 : Dense(64)+LeakyReLU  -> Dense(32)+Tanh  -> Output
  LSTM2 : Dense(64)+ELU        -> Dense(32)+Tanh  -> Output
  LSTM3 : Dense(64)+ReLU       -> Dense(32)+ELU   -> Output

Backend: Pure PyTorch with CUDA (RTX 5080 / cuDNN)
=============================================================================
"""

import os, re, time
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix
)

# CONFIG
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
LEAKY_ALPHA  = 0.01
ELU_ALPHA    = 1.0
RESULTS_DIR  = "results"

os.makedirs(RESULTS_DIR, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("PyTorch  :", torch.__version__)
print("Device   :", DEVICE)
if DEVICE.type == "cuda":
    print("GPU      :", torch.cuda.get_device_name(0))
    print("VRAM     :", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1), "GB")

# DATA
print("\n[1/5] Loading IMDb CSV dataset ...")

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
print(f"  Positive ratio train: {y_train_arr.mean():.2%}  test: {y_test_arr.mean():.2%}")

class IMDbDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def make_loaders(X_tr, y_tr, X_te, y_te, val_split=0.1):
    n_val = int(len(X_tr) * val_split)
    train_ds = IMDbDataset(X_tr[n_val:], y_tr[n_val:])
    val_ds   = IMDbDataset(X_tr[:n_val], y_tr[:n_val])
    test_ds  = IMDbDataset(X_te, y_te)
    kw = dict(batch_size=BATCH_SIZE, num_workers=0, pin_memory=(DEVICE.type=="cuda"))
    return (DataLoader(train_ds, shuffle=True, **kw),
            DataLoader(val_ds,   shuffle=False, **kw),
            DataLoader(test_ds,  shuffle=False, **kw))

# MODELS
print("\n[2/5] Building models ...")

class SentimentLSTM(nn.Module):
    def __init__(self, name, act1, act2):
        super().__init__()
        self.model_name = name
        self.embedding  = nn.Embedding(VOCAB_SIZE, EMBED_DIM, padding_idx=0)
        self.bilstm     = nn.LSTM(EMBED_DIM, LSTM1_UNITS,
                                  batch_first=True, bidirectional=True,
                                  dropout=DROPOUT_RATE)
        self.dropout1   = nn.Dropout(DROPOUT_RATE)
        self.lstm       = nn.LSTM(LSTM1_UNITS * 2, LSTM2_UNITS, batch_first=True)
        self.dropout2   = nn.Dropout(DROPOUT_RATE)
        self.dense1     = nn.Linear(LSTM2_UNITS, DENSE1_UNITS)
        self.act1       = act1
        self.dense2     = nn.Linear(DENSE1_UNITS, DENSE2_UNITS)
        self.act2       = act2
        self.output     = nn.Linear(DENSE2_UNITS, 1)

    def forward(self, x):
        e = self.embedding(x)
        o, _ = self.bilstm(e)
        o = self.dropout1(o)
        _, (h, _) = self.lstm(o)
        o = self.dropout2(h[-1])
        o = self.act1(self.dense1(o))
        o = self.act2(self.dense2(o))
        return self.output(o).squeeze(1)

def build_model1():
    return SentimentLSTM("LSTM1_LeakyReLU_Tanh", nn.LeakyReLU(LEAKY_ALPHA), nn.Tanh()).to(DEVICE)

def build_model2():
    return SentimentLSTM("LSTM2_ELU_Tanh", nn.ELU(ELU_ALPHA), nn.Tanh()).to(DEVICE)

def build_model3():
    return SentimentLSTM("LSTM3_ReLU_ELU", nn.ReLU(), nn.ELU(ELU_ALPHA)).to(DEVICE)

# TRAIN / EVAL
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * len(y_batch)
        preds = (torch.sigmoid(logits) >= 0.5).long()
        correct += (preds == y_batch.long()).sum().item()
        total += len(y_batch)
    return total_loss / total, correct / total

@torch.no_grad()
def eval_epoch(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        total_loss += loss.item() * len(y_batch)
        preds = (torch.sigmoid(logits) >= 0.5).long()
        correct += (preds == y_batch.long()).sum().item()
        total += len(y_batch)
    return total_loss / total, correct / total

def train_model(model, X_tr, y_tr, X_te, y_te):
    train_loader, val_loader, _ = make_loaders(X_tr, y_tr, X_te, y_te)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-6)
    criterion = nn.BCEWithLogitsLoss()
    history = {"accuracy": [], "val_accuracy": [], "loss": [], "val_loss": []}
    best_val_loss, best_state, patience_counter = float("inf"), None, 0

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion)
        vl_loss, vl_acc = eval_epoch(model, val_loader, criterion)
        scheduler.step(vl_loss)
        elapsed = time.time() - t0
        history["loss"].append(tr_loss);     history["accuracy"].append(tr_acc)
        history["val_loss"].append(vl_loss); history["val_accuracy"].append(vl_acc)
        lr = optimizer.param_groups[0]["lr"]
        print(f"  Epoch {epoch:2d}/{EPOCHS}  loss={tr_loss:.4f}  acc={tr_acc:.4f}  "
              f"val_loss={vl_loss:.4f}  val_acc={vl_acc:.4f}  lr={lr:.2e}  [{elapsed:.1f}s]")
        if vl_loss < best_val_loss:
            best_val_loss = vl_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 3:
                print("  Early stopping.")
                break

    model.load_state_dict(best_state)
    return history

@torch.no_grad()
def evaluate_model(model, X_te, y_te):
    model.eval()
    _, _, test_loader = make_loaders(X_train, y_train_arr, X_te, y_te)
    all_probs, all_preds = [], []
    for X_batch, _ in test_loader:
        logits = model(X_batch.to(DEVICE))
        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.extend(probs.tolist())
        all_preds.extend((probs >= 0.5).astype(int).tolist())
    y_pred = np.array(all_preds); y_prob = np.array(all_probs)
    acc  = accuracy_score(y_te, y_pred)
    prec = precision_score(y_te, y_pred)
    rec  = recall_score(y_te, y_pred)
    f1   = f1_score(y_te, y_pred)
    print(f"\n{'='*55}\n  {model.model_name}\n{'='*55}")
    print(f"  Accuracy : {acc*100:.1f}%  Precision: {prec*100:.1f}%  Recall: {rec*100:.1f}%  F1: {f1*100:.1f}%")
    print(classification_report(y_te, y_pred, target_names=["Negative", "Positive"]))
    return {"name": model.model_name, "accuracy": acc, "precision": prec,
            "recall": rec, "f1": f1, "y_pred": y_pred, "y_prob": y_prob}

# RUN EXPERIMENTS
print("\n[3/5] Training all three models on", DEVICE, "...\n")
builders  = [build_model1, build_model2, build_model3]
histories = []
results   = []

for build_fn in builders:
    model = build_fn()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n{'─'*55}\n  Model : {model.model_name}\n  Params: {total_params:,}\n{'─'*55}")
    hist = train_model(model, X_train, y_train_arr, X_test, y_test_arr)
    histories.append(hist)
    res = evaluate_model(model, X_test, y_test_arr)
    results.append(res)
    torch.save(model.state_dict(), os.path.join(RESULTS_DIR, f"{model.model_name}.pt"))
    del model; torch.cuda.empty_cache()

# PLOTS
print("\n[4/5] Generating plots ...")
colors = ["#4C72B0", "#DD8452", "#55A868"]
labels = [r["name"] for r in results]

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("Training & Validation Curves  (IJARCCE 2025)", fontsize=14, fontweight="bold")
for idx, (hist, color) in enumerate(zip(histories, colors)):
    for row, (metric, ylabel) in enumerate([("accuracy","Accuracy"),("loss","Loss")]):
        ax = axes[row][idx]
        ax.plot(hist[metric], color=color, linewidth=2, label=f"Train {ylabel}")
        ax.plot(hist[f"val_{metric}"], color=color, linewidth=2, linestyle="--", label=f"Val {ylabel}")
        ax.set_title(labels[idx], fontsize=11); ax.set_xlabel("Epoch"); ax.set_ylabel(ylabel)
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "01_training_curves.png"), dpi=150, bbox_inches="tight")
plt.close(); print("  Saved: 01_training_curves.png")

x, width = np.arange(len(labels)), 0.18
fig, ax = plt.subplots(figsize=(12, 6))
for i, metric in enumerate(["accuracy","precision","recall","f1"]):
    vals = [r[metric]*100 for r in results]
    bars = ax.bar(x + i*width, vals, width, label=metric.capitalize(), alpha=0.85)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.15, f"{val:.1f}", ha="center", va="bottom", fontsize=8)
ax.set_xlabel("Model"); ax.set_ylabel("Score (%)")
ax.set_title("Performance Comparison – All Three LSTM Models\n(cf. Table 1, IJARCCE 2025)", fontweight="bold")
ax.set_xticks(x + width*1.5); ax.set_xticklabels(labels, rotation=15, ha="right")
ax.set_ylim(75, 100); ax.legend(); ax.grid(True, axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "02_metric_comparison.png"), dpi=150, bbox_inches="tight")
plt.close(); print("  Saved: 02_metric_comparison.png")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle("Confusion Matrices", fontsize=13, fontweight="bold")
for res, ax in zip(results, axes):
    cm = confusion_matrix(y_test_arr, res["y_pred"])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Neg","Pos"], yticklabels=["Neg","Pos"])
    ax.set_title(res["name"], fontsize=10); ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "03_confusion_matrices.png"), dpi=150, bbox_inches="tight")
plt.close(); print("  Saved: 03_confusion_matrices.png")

metric_names = ["Accuracy","Precision","Recall","F1 Score"]
angles = np.linspace(0, 2*np.pi, 4, endpoint=False).tolist(); angles += angles[:1]
fig, ax = plt.subplots(figsize=(7,7), subplot_kw=dict(polar=True))
ax.set_title("Radar Chart – Model Comparison", fontsize=13, fontweight="bold", pad=20)
for res, color in zip(results, colors):
    vals = [res["accuracy"], res["precision"], res["recall"], res["f1"]]; vals += vals[:1]
    ax.plot(angles, vals, color=color, linewidth=2, label=res["name"])
    ax.fill(angles, vals, color=color, alpha=0.1)
ax.set_xticks(angles[:-1]); ax.set_xticklabels(metric_names, size=11)
ax.set_ylim(0.80, 1.00); ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1))
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "04_radar_chart.png"), dpi=150, bbox_inches="tight")
plt.close(); print("  Saved: 04_radar_chart.png")

paper_ref = {
    "LSTM1_LeakyReLU_Tanh": (88.3, 86.1, 91.7, 88.8),
    "LSTM2_ELU_Tanh":        (88.3, 88.7, 88.0, 88.3),
    "LSTM3_ReLU_ELU":        (88.4, 90.3, 86.3, 88.3),
}
fig, axes = plt.subplots(1, 2, figsize=(14, 3))
fig.suptitle("Table 1: Paper Results vs Our Reproduction", fontsize=13, fontweight="bold")
cols = ["Model","Accuracy","Precision","Recall","F1"]
for ax, (data, title) in zip(axes, [
    ([[k,f"{v[0]}%",f"{v[1]}%",f"{v[2]}%",f"{v[3]}%"] for k,v in paper_ref.items()], "Paper (IJARCCE 2025)"),
    ([[r["name"],f"{r['accuracy']*100:.1f}%",f"{r['precision']*100:.1f}%",
       f"{r['recall']*100:.1f}%",f"{r['f1']*100:.1f}%"] for r in results], "Our Reproduction"),
]):
    t = ax.table(cellText=data, colLabels=cols, loc="center", cellLoc="center")
    t.auto_set_font_size(False); t.set_fontsize(9); t.scale(1, 2.2)
    ax.axis("off"); ax.set_title(title, fontsize=11, pad=8)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "05_table_comparison.png"), dpi=150, bbox_inches="tight")
plt.close(); print("  Saved: 05_table_comparison.png")

# SUMMARY
print("\n[5/5] Final Results (cf. Table 1 in paper)\n")
print(f"{'Model':<28} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
print("-"*62)
for res in results:
    print(f"{res['name']:<28} {res['accuracy']*100:>9.1f}% {res['precision']*100:>9.1f}% {res['recall']*100:>9.1f}% {res['f1']*100:>9.1f}%")
print("\nPaper reference:")
for name,(acc,prec,rec,f1) in paper_ref.items():
    print(f"{name:<28} {acc:>9.1f}% {prec:>9.1f}% {rec:>9.1f}% {f1:>9.1f}%")
rows=[]
for res in results:
    rows.append({"Model":res["name"],"Source":"Ours","Accuracy (%)":round(res["accuracy"]*100,2),
                 "Precision (%)":round(res["precision"]*100,2),"Recall (%)":round(res["recall"]*100,2),"F1 Score (%)":round(res["f1"]*100,2)})
for name,(acc,prec,rec,f1) in paper_ref.items():
    rows.append({"Model":name,"Source":"Paper","Accuracy (%)":acc,"Precision (%)":prec,"Recall (%)":rec,"F1 Score (%)":f1})
pd.DataFrame(rows).to_csv(os.path.join(RESULTS_DIR,"results_summary.csv"),index=False)
print(f"\nAll outputs saved to '{RESULTS_DIR}/'")
print("Done.")

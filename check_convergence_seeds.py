"""
Seed-robustness check for RQ2 / Table VII (convergence speed).

Table VII rests on a single seed (42). Its headline is a 17-point spread in
epoch-1 validation accuracy (Mish 81.08 vs ReLU 63.85). This reruns the same
ten homogeneous GRU pairs under several seeds and logs epoch-1 validation loss
and accuracy, so we can say whether that spread is real or a seed artefact.

PIPELINE FIDELITY -- this matters, and a first attempt got it wrong.
The conditions below replicate convergence_rerun.py EXACTLY:
  * its PaperGRU (dropout= arg on the 1-layer BiGRU, o[:,-1,:] readout)
  * its seeding: ONLY torch.manual_seed(s) + np.random.seed(s)
  * NO cudnn.deterministic / cudnn.benchmark flags
Setting cudnn.deterministic=True (as pipeline.py does, correctly, for its own
purpose) perturbs the numerics enough to flip epoch-1 outcomes, because
epoch-1 accuracy is a knife-edge: the model either escapes the ~50% plateau
within one pass or it does not. Verified: with the conditions below, seed 42
reproduces Table VII bit-for-bit (Mish 81.08, GELU 78.00, ReLU 63.85,
Tanh 68.38 -- all delta 0.00).

Output: results/convergence_seeds.csv
"""
import re, csv, time
from collections import Counter
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import warnings; warnings.filterwarnings("ignore")

VOCAB_SIZE, MAX_LEN, EMBED_DIM = 5000, 200, 128
GRU1_UNITS, GRU2_UNITS, DENSE1_UNITS, DENSE2_UNITS = 64, 32, 64, 32
DROPOUT_RATE, BATCH_SIZE = 0.4, 64
SEEDS = [42, 0, 1, 2, 3]
AFS = ["ReLU","ELU","LeakyReLU","Tanh","GELU","SiLU","Mish","SELU","PReLU","Hardswish"]
OUT = "results/convergence_seeds.csv"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Mish(nn.Module):
    def forward(self, x): return x * torch.tanh(F.softplus(x))
ACT = {"ReLU":lambda:nn.ReLU(), "ELU":lambda:nn.ELU(alpha=1.0),
       "LeakyReLU":lambda:nn.LeakyReLU(0.01), "Tanh":lambda:nn.Tanh(),
       "GELU":lambda:nn.GELU(), "SiLU":lambda:nn.SiLU(), "Mish":lambda:Mish(),
       "SELU":lambda:nn.SELU(), "PReLU":lambda:nn.PReLU(), "Hardswish":lambda:nn.Hardswish()}

class PaperGRU(nn.Module):                      # verbatim from convergence_rerun.py
    def __init__(s, a1, a2):
        super().__init__()
        s.embedding = nn.Embedding(VOCAB_SIZE, EMBED_DIM, padding_idx=0)
        s.bigru = nn.GRU(EMBED_DIM, GRU1_UNITS, batch_first=True,
                         bidirectional=True, dropout=DROPOUT_RATE)
        s.gru = nn.GRU(GRU1_UNITS*2, GRU2_UNITS, batch_first=True)
        s.drop = nn.Dropout(DROPOUT_RATE)
        s.dense1 = nn.Linear(GRU2_UNITS, DENSE1_UNITS); s.act1 = ACT[a1]()
        s.dense2 = nn.Linear(DENSE1_UNITS, DENSE2_UNITS); s.act2 = ACT[a2]()
        s.output = nn.Linear(DENSE2_UNITS, 1)
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
    Xtr_t, _, ytr, _ = train_test_split(df["review"].values, df["sentiment"].values,
        test_size=0.20, random_state=42, stratify=df["sentiment"].values)
    c = Counter()
    for t in Xtr_t: c.update(t.split())
    w2i = {"<PAD>":0, "<UNK>":1}
    for w, _ in c.most_common(VOCAB_SIZE-2): w2i[w] = len(w2i)
    out = np.zeros((len(Xtr_t), MAX_LEN), dtype=np.int64)
    for i, t in enumerate(Xtr_t):
        ids = [w2i.get(x, 1) for x in t.split()][:MAX_LEN]; out[i, :len(ids)] = ids
    return out, ytr

def epoch1(af, seed, Xtr, ytr):
    torch.manual_seed(seed); np.random.seed(seed)      # exactly convergence_rerun.py
    m = PaperGRU(af, af).to(DEVICE)
    nv = int(len(Xtr)*0.1); kw = dict(batch_size=BATCH_SIZE, num_workers=0, pin_memory=True)
    tr = DataLoader(DS(Xtr[nv:], ytr[nv:]), shuffle=True, **kw)
    va = DataLoader(DS(Xtr[:nv], ytr[:nv]), shuffle=False, **kw)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3); crit = nn.BCEWithLogitsLoss()
    m.train()
    for Xb, yb in tr:
        Xb, yb = Xb.to(DEVICE), yb.to(DEVICE); opt.zero_grad()
        loss = crit(m(Xb), yb); loss.backward()
        nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
    m.eval(); vl = cc = tt = 0
    with torch.no_grad():
        for Xb, yb in va:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE); lo = m(Xb)
            vl += crit(lo, yb).item()*len(yb)
            cc += ((torch.sigmoid(lo) >= 0.5).long() == yb.long()).sum().item(); tt += len(yb)
    del m; torch.cuda.empty_cache()
    return round(vl/tt, 4), round(cc/tt*100, 2)

if __name__ == "__main__":
    Xtr, ytr = load()
    t0 = time.time(); n = 0; tot = len(AFS)*len(SEEDS)
    with open(OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["af","seed","val_loss_e1","val_acc_e1"])
        w.writeheader()
        for af in AFS:
            accs = []
            for s in SEEDS:
                l, a = epoch1(af, s, Xtr, ytr)
                w.writerow(dict(af=af, seed=s, val_loss_e1=l, val_acc_e1=a)); fh.flush()
                accs.append(a); n += 1
            eta = (time.time()-t0)/n*(tot-n)
            print(f"[{n:2d}/{tot}] {af:10s} {accs}  ETA {eta/60:.1f}m")
    print(f"\nDone in {(time.time()-t0)/60:.1f} min -> {OUT}")

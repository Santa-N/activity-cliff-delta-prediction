"""
SOTA Comparison on MoleculeACE
==============================
Run traditional + GNN baselines alongside our Delta Pred model.
All on the same pre-defined MoleculeACE splits.

Models:
  1. RF + ECFP (traditional SOTA)
  2. SVM + ECFP
  3. XGBoost + ECFP
  4. MLP + ECFP (= ECFP+MLP from literature)
  5. GIN Baseline (our GNN)
  6. GIN + Resid Only (h_B - h_A)
  7. GIN + Delta Pred (Y_A + delta) <- our best
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GINConv, global_add_pool
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit import RDLogger
import warnings, os, importlib.util, time

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

warnings.filterwarnings("ignore")
RDLogger.logger().setLevel(RDLogger.ERROR)
np.random.seed(42)
torch.manual_seed(42)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FP_DIM = 1024
EMB_DIM = 128
print(f"Device: {DEVICE}")


# ============================================================
# Mol features
# ============================================================
ATOM_FEATURES = {
    'atom_type': list(range(1, 54)), 'degree': [0,1,2,3,4,5],
    'formal_charge': [-2,-1,0,1,2],
    'hybridization': [Chem.rdchem.HybridizationType.SP,
                      Chem.rdchem.HybridizationType.SP2,
                      Chem.rdchem.HybridizationType.SP3],
    'aromatic': [False, True], 'num_hs': [0,1,2,3,4],
}

def one_hot(val, allowed):
    enc = [0]*(len(allowed)+1)
    if val in allowed: enc[allowed.index(val)] = 1
    else: enc[-1] = 1
    return enc

def atom_feats(atom):
    f = []
    for key in ['atom_type','degree','formal_charge','hybridization','aromatic','num_hs']:
        f += one_hot(getattr(atom, {
            'atom_type':'GetAtomicNum','degree':'GetDegree',
            'formal_charge':'GetFormalCharge','hybridization':'GetHybridization',
            'aromatic':'GetIsAromatic','num_hs':'GetTotalNumHs'}[key])(),
            ATOM_FEATURES[key])
    return f

ATOM_FEAT_DIM = sum(len(v)+1 for v in ATOM_FEATURES.values())

def mol_to_graph(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None: return None
    x = torch.FloatTensor([atom_feats(a) for a in mol.GetAtoms()])
    edges = []
    for b in mol.GetBonds():
        i,j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        edges += [[i,j],[j,i]]
    ei = torch.LongTensor(edges).t().contiguous() if edges else torch.zeros((2,0),dtype=torch.long)
    return Data(x=x, edge_index=ei)

def smiles_to_fp(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None: return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=FP_DIM)
    arr = np.zeros(FP_DIM, dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

def tanimoto_sim(a, b):
    a_ = a.astype(bool).astype(float)
    b_ = b.astype(bool).astype(float)
    inter = a_ @ b_.T
    return inter / np.maximum(a_.sum(1,keepdims=True) + b_.sum(1,keepdims=True).T - inter, 1e-8)


# ============================================================
# Data
# ============================================================
def get_data_dir():
    spec = importlib.util.find_spec("MoleculeACE")
    return os.path.join(os.path.dirname(spec.origin), "Data", "benchmark_data")

def prepare_data(name):
    df = pd.read_csv(os.path.join(get_data_dir(), name))
    graphs, fps, ys, cliffs, splits = [], [], [], [], []
    for _, row in df.iterrows():
        g = mol_to_graph(row["smiles"])
        fp = smiles_to_fp(row["smiles"])
        if g is not None and fp is not None:
            graphs.append(g); fps.append(fp); ys.append(row["y"])
            cliffs.append(int(row["cliff_mol"])); splits.append(row["split"])
    fps = np.array(fps); ys = np.array(ys, dtype=np.float32)
    cliffs = np.array(cliffs, dtype=bool); splits = np.array(splits)
    tr, te = splits=="train", splits=="test"
    return ([graphs[i] for i in range(len(graphs)) if tr[i]],
            fps[tr], ys[tr], cliffs[tr],
            [graphs[i] for i in range(len(graphs)) if te[i]],
            fps[te], ys[te], cliffs[te])


# ============================================================
# Neighbor retrieval
# ============================================================
def get_neighbors(fps_q, fps_db, y_db, k=3, exclude_self=False):
    sim = tanimoto_sim(fps_q, fps_db)
    if exclude_self: np.fill_diagonal(sim, -1)
    n = len(fps_q)
    h_nb = np.zeros((n, fps_db.shape[1] if hasattr(fps_db, 'shape') else FP_DIM), dtype=np.float32)
    y_nb = np.zeros(n, dtype=np.float32)
    for i in range(n):
        topk = np.argsort(-sim[i])[:k]
        w = sim[i][topk]; w = w / (w.sum() + 1e-8)
        # For embedding-based neighbors, this is called separately
        y_nb[i] = (y_db[topk] * w).sum()
    return y_nb


def get_emb_neighbors(fps_q, fps_db, h_db, y_db, k=3, exclude_self=False):
    sim = tanimoto_sim(fps_q, fps_db)
    if exclude_self: np.fill_diagonal(sim, -1)
    n = len(fps_q)
    h_nb = torch.zeros(n, h_db.shape[1])
    y_nb = np.zeros(n, dtype=np.float32)
    for i in range(n):
        topk = np.argsort(-sim[i])[:k]
        w = sim[i][topk]; w = w / (w.sum() + 1e-8)
        wt = torch.FloatTensor(w)
        h_nb[i] = (h_db[topk] * wt.unsqueeze(-1)).sum(0)
        y_nb[i] = (y_db[topk] * w).sum()
    return h_nb, y_nb


# ============================================================
# GNN Models
# ============================================================
class GINEncoder(nn.Module):
    def __init__(self, in_dim, h=EMB_DIM, n_layers=3, drop=0.2):
        super().__init__()
        self.convs, self.bns = nn.ModuleList(), nn.ModuleList()
        for i in range(n_layers):
            d = in_dim if i==0 else h
            self.convs.append(GINConv(nn.Sequential(nn.Linear(d,h),nn.ReLU(),nn.Linear(h,h))))
            self.bns.append(nn.BatchNorm1d(h))
        self.drop = drop
    def forward(self, data):
        x, ei, batch = data.x, data.edge_index, data.batch
        for c, b in zip(self.convs, self.bns):
            x = F.dropout(F.relu(b(c(x, ei))), self.drop, self.training)
        return global_add_pool(x, batch)

class BaselineGNN(nn.Module):
    def __init__(self, in_dim, h=EMB_DIM):
        super().__init__()
        self.encoder = GINEncoder(in_dim, h)
        self.head = nn.Sequential(nn.Linear(h,h//2),nn.ReLU(),nn.Dropout(0.2),nn.Linear(h//2,1))
    def forward(self, data): return self.head(self.encoder(data)).squeeze(-1)
    def encode(self, data): return self.encoder(data)

class ResidOnlyNet(nn.Module):
    def __init__(self, emb=EMB_DIM):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(emb*2+1,emb),nn.ReLU(),nn.Dropout(0.3),
                                  nn.Linear(emb,emb//2),nn.ReLU(),nn.Dropout(0.3),nn.Linear(emb//2,1))
    def forward(self, h_b, h_a, m_b):
        return self.net(torch.cat([h_b, h_b-h_a, m_b.unsqueeze(-1)],-1)).squeeze(-1)

class DeltaPredNet(nn.Module):
    def __init__(self, emb=EMB_DIM):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(emb*3,emb),nn.ReLU(),nn.Dropout(0.3),
                                  nn.Linear(emb,emb//2),nn.ReLU(),nn.Dropout(0.3),nn.Linear(emb//2,1))
    def forward(self, h_b, h_a, y_a):
        return y_a + self.net(torch.cat([h_b, h_a, h_b-h_a],-1)).squeeze(-1)


# ============================================================
# Training
# ============================================================
def train_gnn(g_tr, y_tr, g_va, y_va, epochs=150, lr=1e-3, pat=25, bs=64):
    model = BaselineGNN(ATOM_FEAT_DIM).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=10, factor=0.5, min_lr=1e-5)
    va_b = Batch.from_data_list(g_va).to(DEVICE)
    va_y = torch.FloatTensor(y_va).to(DEVICE)
    best, bst, w = float("inf"), None, 0
    n = len(g_tr)
    for _ in range(epochs):
        model.train()
        perm = np.random.permutation(n)
        for s in range(0, n, bs):
            idx = perm[s:s+bs]
            b = Batch.from_data_list([g_tr[i] for i in idx]).to(DEVICE)
            yt = torch.FloatTensor(y_tr[idx]).to(DEVICE)
            loss = nn.MSELoss()(model(b), yt)
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad(): vl = nn.MSELoss()(model(va_b), va_y).item()
        sched.step(vl)
        if vl < best: best, bst, w = vl, {k:v.clone() for k,v in model.state_dict().items()}, 0
        else:
            w += 1
            if w >= pat: break
    model.load_state_dict(bst); return model

def get_embeddings(model, graphs, bs=256):
    model.eval(); embs = []
    for s in range(0, len(graphs), bs):
        b = Batch.from_data_list(graphs[s:s+bs]).to(DEVICE)
        with torch.no_grad(): embs.append(model.encode(b).cpu())
    return torch.cat(embs, 0)

def train_correction(net, h_tr, y_tr, h_nb_tr, m_tr, h_va, y_va, h_nb_va, m_va,
                      epochs=200, lr=5e-4, pat=25, bs=64):
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=10, factor=0.5, min_lr=1e-5)
    ds = TensorDataset(h_tr, torch.FloatTensor(y_tr), h_nb_tr, torch.FloatTensor(m_tr))
    loader = DataLoader(ds, batch_size=bs, shuffle=True)
    H_v, Y_v = h_va.to(DEVICE), torch.FloatTensor(y_va).to(DEVICE)
    HN_v, M_v = h_nb_va.to(DEVICE), torch.FloatTensor(m_va).to(DEVICE)
    best, bst, w = float("inf"), None, 0
    for _ in range(epochs):
        net.train()
        for hb, yb, hnb, mb in loader:
            hb, yb, hnb, mb = [t.to(DEVICE) for t in [hb, yb, hnb, mb]]
            loss = nn.MSELoss()(mb + net(hb, hnb, mb), yb)
            opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            vl = nn.MSELoss()(M_v + net(H_v, HN_v, M_v), Y_v).item()
        sched.step(vl)
        if vl < best: best, bst, w = vl, {k:v.clone() for k,v in net.state_dict().items()}, 0
        else:
            w += 1
            if w >= pat: break
    net.load_state_dict(bst); return net

def train_delta(net, h_tr, y_tr, h_nb_tr, y_nb_tr, h_va, y_va, h_nb_va, y_nb_va,
                epochs=200, lr=5e-4, pat=25, bs=64):
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=10, factor=0.5, min_lr=1e-5)
    ds = TensorDataset(h_tr, torch.FloatTensor(y_tr), h_nb_tr, torch.FloatTensor(y_nb_tr))
    loader = DataLoader(ds, batch_size=bs, shuffle=True)
    H_v, Y_v = h_va.to(DEVICE), torch.FloatTensor(y_va).to(DEVICE)
    HN_v, YN_v = h_nb_va.to(DEVICE), torch.FloatTensor(y_nb_va).to(DEVICE)
    best, bst, w = float("inf"), None, 0
    for _ in range(epochs):
        net.train()
        for hb, yb, hnb, ynb in loader:
            hb, yb, hnb, ynb = [t.to(DEVICE) for t in [hb, yb, hnb, ynb]]
            loss = nn.MSELoss()(net(hb, hnb, ynb), yb)
            opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            vl = nn.MSELoss()(net(H_v, HN_v, YN_v), Y_v).item()
        sched.step(vl)
        if vl < best: best, bst, w = vl, {k:v.clone() for k,v in net.state_dict().items()}, 0
        else:
            w += 1
            if w >= pat: break
    net.load_state_dict(bst); return net


# ============================================================
# MLP baseline (ECFP+MLP)
# ============================================================
class MLPBaseline(nn.Module):
    def __init__(self, dim=FP_DIM, h=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim,h),nn.ReLU(),nn.Dropout(0.2),
                                  nn.Linear(h,h//2),nn.ReLU(),nn.Dropout(0.2),nn.Linear(h//2,1))
    def forward(self, x): return self.net(x).squeeze(-1)

def train_mlp(fps_tr, y_tr, fps_va, y_va, epochs=300, lr=1e-3, pat=30, bs=64):
    model = MLPBaseline().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=10, factor=0.5, min_lr=1e-5)
    X_tr, Y_tr = torch.FloatTensor(fps_tr).to(DEVICE), torch.FloatTensor(y_tr).to(DEVICE)
    X_va, Y_va = torch.FloatTensor(fps_va).to(DEVICE), torch.FloatTensor(y_va).to(DEVICE)
    ds = TensorDataset(X_tr, Y_tr)
    loader = DataLoader(ds, batch_size=bs, shuffle=True)
    best, bst, w = float("inf"), None, 0
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            loss = nn.MSELoss()(model(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad(): vl = nn.MSELoss()(model(X_va), Y_va).item()
        sched.step(vl)
        if vl < best: best, bst, w = vl, {k:v.clone() for k,v in model.state_dict().items()}, 0
        else:
            w += 1
            if w >= pat: break
    model.load_state_dict(bst); return model


# ============================================================
# Run one dataset
# ============================================================
def run_dataset(name):
    (g_tr, fps_tr, y_tr, cliff_tr,
     g_te, fps_te, y_te, cliff_te) = prepare_data(name)
    if len(g_tr) < 20 or len(g_te) < 5: return None

    n = len(g_tr)
    perm = np.random.permutation(n)
    nv = max(int(n*0.15), 10)
    vi, ti = perm[:nv], perm[nv:]

    g_va = [g_tr[i] for i in vi]; fps_va, y_va = fps_tr[vi], y_tr[vi]
    g_tr2 = [g_tr[i] for i in ti]; fps_tr2, y_tr2 = fps_tr[ti], y_tr[ti]

    results = {}

    # ---- 1. RF + ECFP ----
    rf = RandomForestRegressor(n_estimators=500, max_depth=None, min_samples_leaf=2,
                                n_jobs=-1, random_state=42)
    rf.fit(fps_tr, y_tr)  # use full train (no val needed)
    results["RF_ECFP"] = rf.predict(fps_te)

    # ---- 2. SVM + ECFP ----
    svm = SVR(kernel='rbf', C=10, gamma='scale')
    svm.fit(fps_tr, y_tr)
    results["SVM_ECFP"] = svm.predict(fps_te)

    # ---- 3. XGBoost + ECFP ----
    if HAS_XGB:
        xgb = XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.1,
                            subsample=0.8, random_state=42, verbosity=0)
        xgb.fit(fps_tr, y_tr)
        results["XGB_ECFP"] = xgb.predict(fps_te)

    # ---- 4. MLP + ECFP ----
    mlp = train_mlp(fps_tr2, y_tr2, fps_va, y_va)
    mlp.eval()
    with torch.no_grad():
        results["MLP_ECFP"] = mlp(torch.FloatTensor(fps_te).to(DEVICE)).cpu().numpy()

    # ---- 5. GIN Baseline ----
    gin = train_gnn(g_tr2, y_tr2, g_va, y_va)
    gin.eval()
    with torch.no_grad():
        results["GIN_Base"] = gin(Batch.from_data_list(g_te).to(DEVICE)).cpu().numpy()

    # Get embeddings for residual models
    h_tr = get_embeddings(gin, g_tr2)
    h_va = get_embeddings(gin, g_va)
    h_te = get_embeddings(gin, g_te)
    h_all = get_embeddings(gin, g_tr)

    with torch.no_grad():
        m_tr = gin(Batch.from_data_list(g_tr2).to(DEVICE)).cpu().numpy()
        m_va = gin(Batch.from_data_list(g_va).to(DEVICE)).cpu().numpy()
        m_te = gin(Batch.from_data_list(g_te).to(DEVICE)).cpu().numpy()

    # Neighbors
    h_nb_tr, y_nb_tr = get_emb_neighbors(fps_tr2, fps_tr2, h_tr, y_tr2, exclude_self=True)
    h_nb_va, y_nb_va = get_emb_neighbors(fps_va, fps_tr2, h_tr, y_tr2)
    h_nb_te, y_nb_te = get_emb_neighbors(fps_te, fps_tr, h_all, y_tr)

    # ---- 6. GIN + Resid Only ----
    rnet = ResidOnlyNet().to(DEVICE)
    rnet = train_correction(rnet, h_tr, y_tr2, h_nb_tr, m_tr, h_va, y_va, h_nb_va, m_va)
    rnet.eval()
    with torch.no_grad():
        p = torch.FloatTensor(m_te).to(DEVICE) + rnet(
            h_te.to(DEVICE), h_nb_te.to(DEVICE), torch.FloatTensor(m_te).to(DEVICE))
        results["GIN_ResidOnly"] = p.cpu().numpy()

    # ---- 7. GIN + Delta Pred ----
    dnet = DeltaPredNet().to(DEVICE)
    dnet = train_delta(dnet, h_tr, y_tr2, h_nb_tr, y_nb_tr, h_va, y_va, h_nb_va, y_nb_va)
    dnet.eval()
    with torch.no_grad():
        p = dnet(h_te.to(DEVICE), h_nb_te.to(DEVICE),
                  torch.FloatTensor(y_nb_te).to(DEVICE))
        results["GIN_DeltaPred"] = p.cpu().numpy()

    # Metrics
    def calc(yt, yp):
        return np.sqrt(mean_squared_error(yt, yp))

    out = {"dataset": name, "n_test": len(y_te), "n_cliff": int(cliff_te.sum()), "metrics": {}}
    for rn, pred in results.items():
        ov = calc(y_te, pred)
        cl = calc(y_te[cliff_te], pred[cliff_te]) if cliff_te.sum() > 1 else np.nan
        out["metrics"][rn] = {"overall_rmse": ov, "cliff_rmse": cl}
    return out


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 100)
    print("  SOTA Comparison on MoleculeACE (30 datasets)")
    print("=" * 100)

    data_dir = get_data_dir()
    datasets = sorted([f for f in os.listdir(data_dir) if f.endswith(".csv")])

    all_res = []
    for i, d in enumerate(datasets):
        print(f"\n[{i+1}/{len(datasets)}] {d}...", end=" ", flush=True)
        t0 = time.time()
        try:
            r = run_dataset(d)
            if r is None: print("SKIP"); continue
            all_res.append(r)
            m = r["metrics"]
            print(f"({time.time()-t0:.1f}s) "
                  f"RF={m['RF_ECFP']['overall_rmse']:.3f} "
                  f"SVM={m['SVM_ECFP']['overall_rmse']:.3f} "
                  f"MLP={m['MLP_ECFP']['overall_rmse']:.3f} "
                  f"GIN={m['GIN_Base']['overall_rmse']:.3f} "
                  f"Delta={m['GIN_DeltaPred']['overall_rmse']:.3f}")
        except Exception as e:
            print(f"ERROR: {e}")
            import traceback; traceback.print_exc()

    # ============================================================
    # Results
    # ============================================================
    keys = ["RF_ECFP", "SVM_ECFP"]
    if HAS_XGB: keys.append("XGB_ECFP")
    keys += ["MLP_ECFP", "GIN_Base", "GIN_ResidOnly", "GIN_DeltaPred"]

    labels = {
        "RF_ECFP": "RF + ECFP",
        "SVM_ECFP": "SVM + ECFP",
        "XGB_ECFP": "XGBoost + ECFP",
        "MLP_ECFP": "MLP + ECFP",
        "GIN_Base": "GIN Baseline",
        "GIN_ResidOnly": "GIN + Resid Only",
        "GIN_DeltaPred": "GIN + Delta Pred (ours)",
    }

    print("\n\n" + "=" * 100)
    print("  OVERALL RESULTS (Mean RMSE across 30 datasets)")
    print("=" * 100)
    print(f"\n{'Model':35s} | {'Overall RMSE':>12s} | {'Cliff RMSE':>10s} | {'Win Overall':>11s} | {'Win Cliff':>10s}")
    print("-" * 90)

    for key in keys:
        rmses = [r["metrics"][key]["overall_rmse"] for r in all_res]
        cliffs = [r["metrics"][key]["cliff_rmse"] for r in all_res
                  if not np.isnan(r["metrics"][key]["cliff_rmse"])]
        # Count wins (best overall among all models for each dataset)
        ov_wins = 0
        cl_wins = 0
        for r in all_res:
            all_ov = {k: r["metrics"][k]["overall_rmse"] for k in keys if k in r["metrics"]}
            if key == min(all_ov, key=all_ov.get): ov_wins += 1
            all_cl = {k: r["metrics"][k]["cliff_rmse"] for k in keys
                      if k in r["metrics"] and not np.isnan(r["metrics"][k]["cliff_rmse"])}
            if all_cl and key == min(all_cl, key=all_cl.get): cl_wins += 1

        print(f"{labels.get(key, key):35s} | {np.mean(rmses):12.4f} | "
              f"{np.mean(cliffs):10.4f} | {ov_wins:>11d} | {cl_wins:>10d}")

    # Per-dataset table
    short = {"RF_ECFP": "RF", "SVM_ECFP": "SVM", "XGB_ECFP": "XGB",
             "MLP_ECFP": "MLP", "GIN_Base": "GIN", "GIN_ResidOnly": "ResO",
             "GIN_DeltaPred": "Delta"}

    print(f"\n{'':28s}", end="")
    for k in keys:
        print(f" | {short.get(k,k):>6s}", end="")
    print()
    print("-" * 95)

    for r in all_res:
        m = r["metrics"]
        vals = [m[k]["overall_rmse"] for k in keys]
        bv = min(vals)
        line = f"{r['dataset']:28s}"
        for v in vals:
            mk = "*" if abs(v-bv) < 1e-6 else " "
            line += f" | {v:5.3f}{mk}"
        print(line)

    means = [np.mean([r["metrics"][k]["overall_rmse"] for r in all_res]) for k in keys]
    bm = min(means)
    line = f"{'MEAN':28s}"
    for v in means:
        mk = "*" if abs(v-bm) < 1e-6 else " "
        line += f" | {v:5.3f}{mk}"
    print("-" * 95)
    print(line)

    # Cliff subset
    print(f"\n[Cliff Subset RMSE]")
    print(f"{'':28s}", end="")
    for k in keys:
        print(f" | {short.get(k,k):>6s}", end="")
    print()
    print("-" * 95)

    for r in all_res:
        m = r["metrics"]
        vals = [m[k]["cliff_rmse"] for k in keys]
        valid = [v for v in vals if not np.isnan(v)]
        if not valid: continue
        bv = min(valid)
        line = f"{r['dataset']:28s}"
        for v in vals:
            if np.isnan(v): line += f" |   nan "
            else:
                mk = "*" if abs(v-bv) < 1e-6 else " "
                line += f" | {v:5.3f}{mk}"
        print(line)

    cmeans = [np.mean([r["metrics"][k]["cliff_rmse"] for r in all_res
                       if not np.isnan(r["metrics"][k]["cliff_rmse"])]) for k in keys]
    bcm = min(cmeans)
    line = f"{'MEAN':28s}"
    for v in cmeans:
        mk = "*" if abs(v-bcm) < 1e-6 else " "
        line += f" | {v:5.3f}{mk}"
    print("-" * 95)
    print(line)

    # Literature comparison
    print("\n\n" + "=" * 100)
    print("  COMPARISON WITH PUBLISHED RESULTS (from REMO paper, MTPNet paper)")
    print("=" * 100)
    print("""
    Published results (mean RMSE / cliff RMSE across 30 MoleculeACE datasets):
    +---------------------------------+-----------+------------+
    | Model                           |    RMSE   | Cliff RMSE |
    +---------------------------------+-----------+------------+
    | ECFP+SVM (published)            |    0.675  |    0.762   |
    | ECFP+MLP (published)            |    0.772  |    0.831   |
    | D-MPNN (published)              |    0.750  |    0.829   |
    | GIN (published)                 |    0.815  |    0.901   |
    | Graphormer (published)          |    0.805  |    0.848   |
    | REMOIM-Graphormer (SOTA, pub.)  |    0.647  |    0.747   |
    +---------------------------------+-----------+------------+""")

    print(f"""
    Our results (same 30 datasets, same splits):
    +---------------------------------+-----------+------------+
    | Model                           |    RMSE   | Cliff RMSE |
    +---------------------------------+-----------+------------+""")
    for key in keys:
        rmses = [r["metrics"][key]["overall_rmse"] for r in all_res]
        cliffs = [r["metrics"][key]["cliff_rmse"] for r in all_res
                  if not np.isnan(r["metrics"][key]["cliff_rmse"])]
        print(f"    | {labels.get(key,key):31s} | {np.mean(rmses):9.4f} | {np.mean(cliffs):10.4f} |")
    print(f"    +---------------------------------+-----------+------------+")


if __name__ == "__main__":
    main()

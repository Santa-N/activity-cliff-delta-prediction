"""
Universal Delta Prediction: Does it improve EVERY model?
========================================================
For each base model, compare:
  - Base: model(B) -> Y
  - +Delta: Y_A + model(B, A, B-A) -> Y

Models: RF, SVM, XGBoost, MLP, GIN
All on MoleculeACE 30 datasets.
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
# Mol features (same as before)
# ============================================================
ATOM_FEATURES = {
    'atom_type': list(range(1,54)), 'degree': [0,1,2,3,4,5],
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
        f += one_hot(getattr(atom, {'atom_type':'GetAtomicNum','degree':'GetDegree',
            'formal_charge':'GetFormalCharge','hybridization':'GetHybridization',
            'aromatic':'GetIsAromatic','num_hs':'GetTotalNumHs'}[key])(), ATOM_FEATURES[key])
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
    a_ = a.astype(bool).astype(float); b_ = b.astype(bool).astype(float)
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
        g = mol_to_graph(row["smiles"]); fp = smiles_to_fp(row["smiles"])
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
# Neighbors
# ============================================================
def get_fp_neighbors(fps_q, fps_db, y_db, k=3, exclude_self=False):
    sim = tanimoto_sim(fps_q, fps_db)
    if exclude_self: np.fill_diagonal(sim, -1)
    n = len(fps_q)
    nb_fps = np.zeros((n, FP_DIM), dtype=np.float32)
    nb_ys = np.zeros(n, dtype=np.float32)
    for i in range(n):
        topk = np.argsort(-sim[i])[:k]
        w = sim[i][topk]; w = w / (w.sum() + 1e-8)
        nb_fps[i] = (fps_db[topk] * w[:, None]).sum(0)
        nb_ys[i] = (y_db[topk] * w).sum()
    return nb_fps, nb_ys

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
# Traditional models: Base vs Delta
# ============================================================
def rf_base(fps_tr, y_tr, fps_te):
    m = RandomForestRegressor(n_estimators=500, min_samples_leaf=2, n_jobs=-1, random_state=42)
    m.fit(fps_tr, y_tr)
    return m.predict(fps_te)

def rf_delta(fps_tr, y_tr, nb_fps_tr, nb_y_tr, fps_te, nb_fps_te, nb_y_te):
    """RF predicts delta: Y(B)-Y(A) from FP(B)-FP(A) features"""
    delta_y_tr = y_tr - nb_y_tr
    diff_tr = fps_tr - nb_fps_tr
    # Features: FP(B), FP(A), FP(B)-FP(A)
    X_tr = np.concatenate([fps_tr, nb_fps_tr, diff_tr], axis=1)
    m = RandomForestRegressor(n_estimators=500, min_samples_leaf=2, n_jobs=-1, random_state=42)
    m.fit(X_tr, delta_y_tr)
    diff_te = fps_te - nb_fps_te
    X_te = np.concatenate([fps_te, nb_fps_te, diff_te], axis=1)
    return nb_y_te + m.predict(X_te)

def svm_base(fps_tr, y_tr, fps_te):
    m = SVR(kernel='rbf', C=10, gamma='scale')
    m.fit(fps_tr, y_tr)
    return m.predict(fps_te)

def svm_delta(fps_tr, y_tr, nb_fps_tr, nb_y_tr, fps_te, nb_fps_te, nb_y_te):
    delta_y_tr = y_tr - nb_y_tr
    diff_tr = fps_tr - nb_fps_tr
    X_tr = np.concatenate([fps_tr, nb_fps_tr, diff_tr], axis=1)
    m = SVR(kernel='rbf', C=10, gamma='scale')
    m.fit(X_tr, delta_y_tr)
    diff_te = fps_te - nb_fps_te
    X_te = np.concatenate([fps_te, nb_fps_te, diff_te], axis=1)
    return nb_y_te + m.predict(X_te)

def xgb_base(fps_tr, y_tr, fps_te):
    m = XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.1,
                      subsample=0.8, random_state=42, verbosity=0)
    m.fit(fps_tr, y_tr)
    return m.predict(fps_te)

def xgb_delta(fps_tr, y_tr, nb_fps_tr, nb_y_tr, fps_te, nb_fps_te, nb_y_te):
    delta_y_tr = y_tr - nb_y_tr
    diff_tr = fps_tr - nb_fps_tr
    X_tr = np.concatenate([fps_tr, nb_fps_tr, diff_tr], axis=1)
    m = XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.1,
                      subsample=0.8, random_state=42, verbosity=0)
    m.fit(X_tr, delta_y_tr)
    diff_te = fps_te - nb_fps_te
    X_te = np.concatenate([fps_te, nb_fps_te, diff_te], axis=1)
    return nb_y_te + m.predict(X_te)


# ============================================================
# MLP: Base vs Delta
# ============================================================
class MLPBase(nn.Module):
    def __init__(self, dim=FP_DIM, h=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim,h),nn.ReLU(),nn.Dropout(0.2),
                                  nn.Linear(h,h//2),nn.ReLU(),nn.Dropout(0.2),nn.Linear(h//2,1))
    def forward(self, x): return self.net(x).squeeze(-1)

class MLPDelta(nn.Module):
    def __init__(self, dim=FP_DIM, h=512):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim*3, h),nn.ReLU(),nn.Dropout(0.3),
                                  nn.Linear(h,h//2),nn.ReLU(),nn.Dropout(0.3),nn.Linear(h//2,1))
    def forward(self, fp_b, fp_a, y_a):
        return y_a + self.net(torch.cat([fp_b, fp_a, fp_b-fp_a],-1)).squeeze(-1)


def train_nn(model, loader, val_fn, epochs=300, lr=1e-3, pat=30):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=10, factor=0.5, min_lr=1e-5)
    best, bst, w = float("inf"), None, 0
    for _ in range(epochs):
        model.train()
        for batch in loader:
            batch = [t.to(DEVICE) for t in batch]
            if len(batch) == 2:
                pred = model(batch[0])
            else:
                pred = model(batch[0], batch[2], batch[3])
            loss = nn.MSELoss()(pred, batch[1])
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad(): vl = val_fn()
        sched.step(vl)
        if vl < best: best, bst, w = vl, {k:v.clone() for k,v in model.state_dict().items()}, 0
        else:
            w += 1
            if w >= pat: break
    model.load_state_dict(bst); return model


# ============================================================
# GIN: Base vs Delta
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

class GINBase(nn.Module):
    def __init__(self, in_dim, h=EMB_DIM):
        super().__init__()
        self.encoder = GINEncoder(in_dim, h)
        self.head = nn.Sequential(nn.Linear(h,h//2),nn.ReLU(),nn.Dropout(0.2),nn.Linear(h//2,1))
    def forward(self, data): return self.head(self.encoder(data)).squeeze(-1)
    def encode(self, data): return self.encoder(data)

class GINDelta(nn.Module):
    def __init__(self, emb=EMB_DIM):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(emb*3,emb),nn.ReLU(),nn.Dropout(0.3),
                                  nn.Linear(emb,emb//2),nn.ReLU(),nn.Dropout(0.3),nn.Linear(emb//2,1))
    def forward(self, h_b, h_a, y_a):
        return y_a + self.net(torch.cat([h_b, h_a, h_b-h_a],-1)).squeeze(-1)


def train_gin(g_tr, y_tr, g_va, y_va, epochs=150, lr=1e-3, pat=25, bs=64):
    model = GINBase(ATOM_FEAT_DIM).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=10, factor=0.5, min_lr=1e-5)
    va_b = Batch.from_data_list(g_va).to(DEVICE); va_y = torch.FloatTensor(y_va).to(DEVICE)
    best, bst, w = float("inf"), None, 0; n = len(g_tr)
    for _ in range(epochs):
        model.train(); perm = np.random.permutation(n)
        for s in range(0, n, bs):
            idx = perm[s:s+bs]
            b = Batch.from_data_list([g_tr[i] for i in idx]).to(DEVICE)
            yt = torch.FloatTensor(y_tr[idx]).to(DEVICE)
            loss = nn.MSELoss()(model(b), yt); opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad(): vl = nn.MSELoss()(model(va_b), va_y).item()
        sched.step(vl)
        if vl < best: best, bst, w = vl, {k:v.clone() for k,v in model.state_dict().items()}, 0
        else:
            w += 1
            if w >= pat: break
    model.load_state_dict(bst); return model

def get_embs(model, graphs, bs=256):
    model.eval(); embs = []
    for s in range(0, len(graphs), bs):
        b = Batch.from_data_list(graphs[s:s+bs]).to(DEVICE)
        with torch.no_grad(): embs.append(model.encode(b).cpu())
    return torch.cat(embs, 0)

def train_gin_delta(net, h_tr, y_tr, h_nb_tr, y_nb_tr, h_va, y_va, h_nb_va, y_nb_va,
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
            loss = nn.MSELoss()(net(hb, hnb, ynb), yb); opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad(): vl = nn.MSELoss()(net(H_v, HN_v, YN_v), Y_v).item()
        sched.step(vl)
        if vl < best: best, bst, w = vl, {k:v.clone() for k,v in net.state_dict().items()}, 0
        else:
            w += 1
            if w >= pat: break
    net.load_state_dict(bst); return net


# ============================================================
# Run one dataset
# ============================================================
def run_dataset(name):
    (g_tr, fps_tr, y_tr, cliff_tr, g_te, fps_te, y_te, cliff_te) = prepare_data(name)
    if len(g_tr) < 20 or len(g_te) < 5: return None

    n = len(g_tr); perm = np.random.permutation(n)
    nv = max(int(n*0.15), 10); vi, ti = perm[:nv], perm[nv:]
    g_va = [g_tr[i] for i in vi]; fps_va, y_va = fps_tr[vi], y_tr[vi]
    g_tr2 = [g_tr[i] for i in ti]; fps_tr2, y_tr2 = fps_tr[ti], y_tr[ti]

    # Neighbors
    nb_fps_tr, nb_y_tr = get_fp_neighbors(fps_tr, fps_tr, y_tr, exclude_self=True)
    nb_fps_te, nb_y_te = get_fp_neighbors(fps_te, fps_tr, y_tr)
    # For val split models
    nb_fps_tr2, nb_y_tr2 = get_fp_neighbors(fps_tr2, fps_tr2, y_tr2, exclude_self=True)
    nb_fps_va, nb_y_va = get_fp_neighbors(fps_va, fps_tr2, y_tr2)

    results = {}

    # ---- RF ----
    results["RF"] = rf_base(fps_tr, y_tr, fps_te)
    results["RF+Delta"] = rf_delta(fps_tr, y_tr, nb_fps_tr, nb_y_tr,
                                    fps_te, nb_fps_te, nb_y_te)

    # ---- SVM ----
    results["SVM"] = svm_base(fps_tr, y_tr, fps_te)
    results["SVM+Delta"] = svm_delta(fps_tr, y_tr, nb_fps_tr, nb_y_tr,
                                      fps_te, nb_fps_te, nb_y_te)

    # ---- XGBoost ----
    if HAS_XGB:
        results["XGB"] = xgb_base(fps_tr, y_tr, fps_te)
        results["XGB+Delta"] = xgb_delta(fps_tr, y_tr, nb_fps_tr, nb_y_tr,
                                          fps_te, nb_fps_te, nb_y_te)

    # ---- MLP ----
    t = lambda x: torch.FloatTensor(x).to(DEVICE)
    mlp = MLPBase().to(DEVICE)
    X_tr, Y_tr = t(fps_tr2), t(y_tr2)
    X_va, Y_va = t(fps_va), t(y_va)
    ds = TensorDataset(X_tr, Y_tr)
    loader = DataLoader(ds, batch_size=64, shuffle=True)
    val_fn = lambda: nn.MSELoss()(mlp(X_va), Y_va).item()
    mlp = train_nn(mlp, loader, val_fn)
    mlp.eval()
    with torch.no_grad(): results["MLP"] = mlp(t(fps_te)).cpu().numpy()

    # MLP + Delta
    mlp_d = MLPDelta().to(DEVICE)
    ds_d = TensorDataset(t(fps_tr2), t(y_tr2), t(nb_fps_tr2), t(nb_y_tr2))
    loader_d = DataLoader(ds_d, batch_size=64, shuffle=True)
    val_fn_d = lambda: nn.MSELoss()(mlp_d(t(fps_va), t(nb_fps_va), t(nb_y_va)), Y_va).item()
    mlp_d = train_nn(mlp_d, loader_d, val_fn_d, lr=5e-4)
    mlp_d.eval()
    with torch.no_grad():
        results["MLP+Delta"] = mlp_d(t(fps_te), t(nb_fps_te), t(nb_y_te)).cpu().numpy()

    # ---- GIN ----
    gin = train_gin(g_tr2, y_tr2, g_va, y_va)
    gin.eval()
    with torch.no_grad():
        results["GIN"] = gin(Batch.from_data_list(g_te).to(DEVICE)).cpu().numpy()

    # GIN + Delta
    h_tr = get_embs(gin, g_tr2); h_va = get_embs(gin, g_va)
    h_te = get_embs(gin, g_te); h_all = get_embs(gin, g_tr)
    h_nb_tr, y_nb_tr = get_emb_neighbors(fps_tr2, fps_tr2, h_tr, y_tr2, exclude_self=True)
    h_nb_va, y_nb_va = get_emb_neighbors(fps_va, fps_tr2, h_tr, y_tr2)
    h_nb_te, y_nb_te = get_emb_neighbors(fps_te, fps_tr, h_all, y_tr)

    gd = GINDelta().to(DEVICE)
    gd = train_gin_delta(gd, h_tr, y_tr2, h_nb_tr, y_nb_tr, h_va, y_va, h_nb_va, y_nb_va)
    gd.eval()
    with torch.no_grad():
        results["GIN+Delta"] = gd(h_te.to(DEVICE), h_nb_te.to(DEVICE),
                                   torch.FloatTensor(y_nb_te).to(DEVICE)).cpu().numpy()

    # Metrics
    out = {"dataset": name, "n_test": len(y_te), "n_cliff": int(cliff_te.sum()), "metrics": {}}
    for rn, pred in results.items():
        ov = np.sqrt(mean_squared_error(y_te, pred))
        cl = np.sqrt(mean_squared_error(y_te[cliff_te], pred[cliff_te])) if cliff_te.sum() > 1 else np.nan
        out["metrics"][rn] = {"overall_rmse": ov, "cliff_rmse": cl}
    return out


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 110)
    print("  Universal Delta Prediction: Base vs +Delta for EVERY model")
    print("=" * 110)

    data_dir = get_data_dir()
    datasets = sorted([f for f in os.listdir(data_dir) if f.endswith(".csv")])
    all_res = []

    for i, d in enumerate(datasets):
        print(f"\n[{i+1}/{len(datasets)}] {d}...", end=" ", flush=True)
        t0 = time.time()
        try:
            r = run_dataset(d)
            if r is None: print("SKIP"); continue
            all_res.append(r); m = r["metrics"]
            print(f"({time.time()-t0:.1f}s) "
                  f"RF={m['RF']['overall_rmse']:.3f}/{m['RF+Delta']['overall_rmse']:.3f} "
                  f"SVM={m['SVM']['overall_rmse']:.3f}/{m['SVM+Delta']['overall_rmse']:.3f} "
                  f"GIN={m['GIN']['overall_rmse']:.3f}/{m['GIN+Delta']['overall_rmse']:.3f}")
        except Exception as e:
            print(f"ERROR: {e}"); import traceback; traceback.print_exc()

    # ============================================================
    # Results: Base vs +Delta comparison
    # ============================================================
    pairs = [("RF", "RF+Delta"), ("SVM", "SVM+Delta")]
    if HAS_XGB: pairs.append(("XGB", "XGB+Delta"))
    pairs += [("MLP", "MLP+Delta"), ("GIN", "GIN+Delta")]

    print("\n\n" + "=" * 110)
    print("  BASE vs +DELTA PREDICTION (Overall RMSE)")
    print("=" * 110)
    print(f"\n{'Model':12s} | {'Base RMSE':>10s} | {'+Delta RMSE':>11s} | {'Improve':>8s} | {'Improve%':>9s} | {'Win':>5s} | {'Cliff Base':>10s} | {'Cliff +D':>9s} | {'Cliff%':>7s}")
    print("-" * 110)

    for base_key, delta_key in pairs:
        base_rmses = [r["metrics"][base_key]["overall_rmse"] for r in all_res]
        delta_rmses = [r["metrics"][delta_key]["overall_rmse"] for r in all_res]
        base_cliff = [r["metrics"][base_key]["cliff_rmse"] for r in all_res
                      if not np.isnan(r["metrics"][base_key]["cliff_rmse"])]
        delta_cliff = [r["metrics"][delta_key]["cliff_rmse"] for r in all_res
                       if not np.isnan(r["metrics"][delta_key]["cliff_rmse"])]
        wins = sum(1 for b, d in zip(base_rmses, delta_rmses) if d < b)
        imp = np.mean([(b-d)/b*100 for b, d in zip(base_rmses, delta_rmses)])
        cimp = np.mean([(b-d)/b*100 for b, d in zip(base_cliff, delta_cliff)])

        print(f"{base_key:12s} | {np.mean(base_rmses):10.4f} | {np.mean(delta_rmses):11.4f} | "
              f"{np.mean(base_rmses)-np.mean(delta_rmses):8.4f} | {imp:+8.2f}% | "
              f"{wins:>2d}/{len(all_res):<2d} | {np.mean(base_cliff):10.4f} | "
              f"{np.mean(delta_cliff):9.4f} | {cimp:+6.2f}%")

    # Per-dataset detail
    print("\n\n" + "=" * 110)
    print("  PER-DATASET: Base / +Delta (Overall RMSE)")
    print("=" * 110)

    header = f"{'':28s}"
    for base_key, _ in pairs:
        header += f" | {base_key:>11s}"
    print(header)

    header2 = f"{'Dataset':28s}"
    for _, _ in pairs:
        header2 += f" | {'Base':>5s} {'->D':>5s}"
    print(header2)
    print("-" * 110)

    for r in all_res:
        m = r["metrics"]
        line = f"{r['dataset']:28s}"
        for base_key, delta_key in pairs:
            b = m[base_key]["overall_rmse"]
            d = m[delta_key]["overall_rmse"]
            marker = "+" if d < b else "-"
            line += f" | {b:5.3f} {d:5.3f}{marker}"
        print(line)

    # Mean
    line = f"{'MEAN':28s}"
    for base_key, delta_key in pairs:
        bm = np.mean([r["metrics"][base_key]["overall_rmse"] for r in all_res])
        dm = np.mean([r["metrics"][delta_key]["overall_rmse"] for r in all_res])
        marker = "+" if dm < bm else "-"
        line += f" | {bm:5.3f} {dm:5.3f}{marker}"
    print("-" * 110)
    print(line)

    # Cliff subset
    print(f"\n[Cliff Subset RMSE]")
    header2 = f"{'Dataset':28s}"
    for _, _ in pairs:
        header2 += f" | {'Base':>5s} {'->D':>5s}"
    print(header2)
    print("-" * 110)

    for r in all_res:
        m = r["metrics"]
        base_cl = m[pairs[0][0]]["cliff_rmse"]
        if np.isnan(base_cl): continue
        line = f"{r['dataset']:28s}"
        for base_key, delta_key in pairs:
            b = m[base_key]["cliff_rmse"]
            d = m[delta_key]["cliff_rmse"]
            if np.isnan(b): line += f" |   nan   nan "
            else:
                marker = "+" if d < b else "-"
                line += f" | {b:5.3f} {d:5.3f}{marker}"
        print(line)

    line = f"{'MEAN':28s}"
    for base_key, delta_key in pairs:
        bm = np.mean([r["metrics"][base_key]["cliff_rmse"] for r in all_res
                       if not np.isnan(r["metrics"][base_key]["cliff_rmse"])])
        dm = np.mean([r["metrics"][delta_key]["cliff_rmse"] for r in all_res
                       if not np.isnan(r["metrics"][delta_key]["cliff_rmse"])])
        marker = "+" if dm < bm else "-"
        line += f" | {bm:5.3f} {dm:5.3f}{marker}"
    print("-" * 110)
    print(line)

    # Summary
    print("\n\n" + "=" * 110)
    print("  SUMMARY: Does Delta Prediction universally improve every model?")
    print("=" * 110)
    for base_key, delta_key in pairs:
        base_rmses = [r["metrics"][base_key]["overall_rmse"] for r in all_res]
        delta_rmses = [r["metrics"][delta_key]["overall_rmse"] for r in all_res]
        wins = sum(1 for b, d in zip(base_rmses, delta_rmses) if d < b)
        imp = np.mean([(b-d)/b*100 for b, d in zip(base_rmses, delta_rmses)])
        print(f"  {base_key:8s} -> +Delta: {imp:+.2f}% mean improvement, "
              f"wins {wins}/{len(all_res)} datasets")


if __name__ == "__main__":
    main()

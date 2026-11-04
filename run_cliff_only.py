"""
All Experiments Re-evaluated on Activity Cliff Molecules ONLY
=============================================================
Training: full dataset (need all molecules for context)
Evaluation: ONLY cliff_mol=1 molecules from test set

Experiments:
1. Universal Delta: Base vs +Delta for every model (RF, SVM, XGB, MLP, GIN)
2. Diagnostic: Why delta fails on strong models (cliff-focused)
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
from scipy.stats import spearmanr
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
# Basics
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

def get_neighbors(fps_q, fps_db, y_db, k=3, exclude_self=False):
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
# Models
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

class MLPDelta(nn.Module):
    def __init__(self, in_dim, h=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim,h),nn.ReLU(),nn.Dropout(0.3),
                                  nn.Linear(h,h//2),nn.ReLU(),nn.Dropout(0.3),nn.Linear(h//2,1))
    def forward(self, x): return self.net(x).squeeze(-1)


# ============================================================
# Training helpers
# ============================================================
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

def train_mlp_delta(in_dim, X_tr, y_tr, X_va, y_va, epochs=300, lr=5e-4, pat=30, bs=64):
    model = MLPDelta(in_dim).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=10, factor=0.5, min_lr=1e-5)
    Xt, Yt = torch.FloatTensor(X_tr).to(DEVICE), torch.FloatTensor(y_tr).to(DEVICE)
    Xv, Yv = torch.FloatTensor(X_va).to(DEVICE), torch.FloatTensor(y_va).to(DEVICE)
    ds = TensorDataset(Xt, Yt); loader = DataLoader(ds, batch_size=bs, shuffle=True)
    best, bst, w = float("inf"), None, 0
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            loss = nn.MSELoss()(model(xb), yb); opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad(): vl = nn.MSELoss()(model(Xv), Yv).item()
        sched.step(vl)
        if vl < best: best, bst, w = vl, {k:v.clone() for k,v in model.state_dict().items()}, 0
        else:
            w += 1
            if w >= pat: break
    model.load_state_dict(bst); return model


# ============================================================
# Run one dataset — evaluate on CLIFF ONLY
# ============================================================
def run_dataset(name):
    (g_tr, fps_tr, y_tr, cliff_tr, g_te, fps_te, y_te, cliff_te) = prepare_data(name)
    if len(g_tr) < 20 or len(g_te) < 5: return None
    if cliff_te.sum() < 3: return None  # need enough cliff molecules

    # Cliff-only test indices
    cliff_idx = np.where(cliff_te)[0]
    fps_te_c = fps_te[cliff_idx]
    y_te_c = y_te[cliff_idx]
    g_te_c = [g_te[i] for i in cliff_idx]

    n = len(g_tr); perm = np.random.permutation(n)
    nv = max(int(n*0.15), 10); vi, ti = perm[:nv], perm[nv:]
    g_va = [g_tr[i] for i in vi]; fps_va, y_va = fps_tr[vi], y_tr[vi]
    g_tr2 = [g_tr[i] for i in ti]; fps_tr2, y_tr2 = fps_tr[ti], y_tr[ti]

    t = lambda x: torch.FloatTensor(x)

    # Neighbors for cliff test molecules (search from full train)
    nb_fps_te, nb_y_te = get_neighbors(fps_te_c, fps_tr, y_tr)
    nb_fps_tr, nb_y_tr = get_neighbors(fps_tr, fps_tr, y_tr, exclude_self=True)
    nb_fps_tr2, nb_y_tr2 = get_neighbors(fps_tr2, fps_tr2, y_tr2, exclude_self=True)
    nb_fps_va, nb_y_va = get_neighbors(fps_va, fps_tr2, y_tr2)

    results = {}

    # ---- Traditional: Base vs Delta ----
    # RF
    rf = RandomForestRegressor(n_estimators=500, min_samples_leaf=2, n_jobs=-1, random_state=42)
    rf.fit(fps_tr, y_tr)
    results["RF"] = rf.predict(fps_te_c)

    rf_d = RandomForestRegressor(n_estimators=500, min_samples_leaf=2, n_jobs=-1, random_state=42)
    X_tr_d = np.concatenate([fps_tr, nb_fps_tr, fps_tr - nb_fps_tr], axis=1)
    rf_d.fit(X_tr_d, y_tr - nb_y_tr)
    X_te_d = np.concatenate([fps_te_c, nb_fps_te, fps_te_c - nb_fps_te], axis=1)
    results["RF+Delta"] = nb_y_te + rf_d.predict(X_te_d)

    # SVM
    svm = SVR(kernel='rbf', C=10, gamma='scale')
    svm.fit(fps_tr, y_tr)
    results["SVM"] = svm.predict(fps_te_c)

    svm_d = SVR(kernel='rbf', C=10, gamma='scale')
    svm_d.fit(X_tr_d, y_tr - nb_y_tr)
    results["SVM+Delta"] = nb_y_te + svm_d.predict(X_te_d)

    # XGBoost
    if HAS_XGB:
        xgb = XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.1,
                            subsample=0.8, random_state=42, verbosity=0)
        xgb.fit(fps_tr, y_tr)
        results["XGB"] = xgb.predict(fps_te_c)

        xgb_d = XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.1,
                              subsample=0.8, random_state=42, verbosity=0)
        xgb_d.fit(X_tr_d, y_tr - nb_y_tr)
        results["XGB+Delta"] = nb_y_te + xgb_d.predict(X_te_d)

    # MLP
    from sklearn.neural_network import MLPRegressor
    mlp = MLPRegressor(hidden_layer_sizes=(256,128), max_iter=500,
                        early_stopping=True, random_state=42)
    mlp.fit(fps_tr, y_tr)
    results["MLP"] = mlp.predict(fps_te_c)

    # MLP+Delta (neural)
    X_tr_mlp = np.concatenate([fps_tr2, nb_fps_tr2, fps_tr2 - nb_fps_tr2], axis=1)
    X_va_mlp = np.concatenate([fps_va, nb_fps_va, fps_va - nb_fps_va], axis=1)
    X_te_mlp = np.concatenate([fps_te_c, nb_fps_te, fps_te_c - nb_fps_te], axis=1)
    mlp_d = train_mlp_delta(X_tr_mlp.shape[1], X_tr_mlp, y_tr2 - nb_y_tr2,
                             X_va_mlp, y_va - nb_y_va)
    mlp_d.eval()
    with torch.no_grad():
        results["MLP+Delta"] = nb_y_te + mlp_d(t(X_te_mlp).to(DEVICE)).cpu().numpy()

    # ---- GIN: Base vs Delta ----
    gin = train_gin(g_tr2, y_tr2, g_va, y_va)
    gin.eval()
    with torch.no_grad():
        results["GIN"] = gin(Batch.from_data_list(g_te_c).to(DEVICE)).cpu().numpy()

    h_tr = get_embs(gin, g_tr2); h_va = get_embs(gin, g_va)
    h_te_c = get_embs(gin, g_te_c); h_all = get_embs(gin, g_tr)
    h_nb_tr, y_nb_tr_e = get_emb_neighbors(fps_tr2, fps_tr2, h_tr, y_tr2, exclude_self=True)
    h_nb_va, y_nb_va_e = get_emb_neighbors(fps_va, fps_tr2, h_tr, y_tr2)
    h_nb_te, y_nb_te_e = get_emb_neighbors(fps_te_c, fps_tr, h_all, y_tr)

    gd = GINDelta().to(DEVICE)
    gd = train_gin_delta(gd, h_tr, y_tr2, h_nb_tr, y_nb_tr_e,
                          h_va, y_va, h_nb_va, y_nb_va_e)
    gd.eval()
    with torch.no_grad():
        results["GIN+Delta"] = gd(h_te_c.to(DEVICE), h_nb_te.to(DEVICE),
                                   t(y_nb_te_e).to(DEVICE)).cpu().numpy()

    # ---- SVM + Neural Correction (on cliff only) ----
    svm_pred_tr2 = svm.predict(fps_tr2)
    svm_pred_va = svm.predict(fps_va)
    svm_pred_te = svm.predict(fps_te_c)

    X_corr_tr = np.concatenate([fps_tr2, nb_fps_tr2, fps_tr2-nb_fps_tr2,
                                 svm_pred_tr2.reshape(-1,1)], axis=1)
    X_corr_va = np.concatenate([fps_va, nb_fps_va, fps_va-nb_fps_va,
                                 svm_pred_va.reshape(-1,1)], axis=1)
    X_corr_te = np.concatenate([fps_te_c, nb_fps_te, fps_te_c-nb_fps_te,
                                 svm_pred_te.reshape(-1,1)], axis=1)

    corr_model = train_mlp_delta(X_corr_tr.shape[1], X_corr_tr, y_tr2 - svm_pred_tr2,
                                  X_corr_va, y_va - svm_pred_va)
    corr_model.eval()
    with torch.no_grad():
        results["SVM+NeuralCorr"] = svm_pred_te + corr_model(
            t(X_corr_te).to(DEVICE)).cpu().numpy()

    # ---- Neighbor Copy baseline ----
    results["NB_Copy"] = nb_y_te

    # ---- Residual analysis ----
    analysis = {}
    for mname, pred in [("SVM", results["SVM"]), ("RF", results["RF"]),
                         ("MLP", results["MLP"]), ("GIN", results["GIN"])]:
        residuals = y_te_c - pred
        corr_ya, _ = spearmanr(residuals, nb_y_te)
        nb_diff = nb_y_te - pred
        corr_nb, _ = spearmanr(residuals, nb_diff)
        analysis[f"{mname}_corr_ya"] = corr_ya
        analysis[f"{mname}_corr_nb"] = corr_nb
        analysis[f"{mname}_resid_std"] = np.std(residuals)

    # Metrics (cliff only)
    out = {"dataset": name, "n_cliff_test": len(y_te_c),
           "metrics": {}, "analysis": analysis}
    for rn, pred in results.items():
        rmse = np.sqrt(mean_squared_error(y_te_c, pred))
        out["metrics"][rn] = {"cliff_rmse": rmse}
    return out


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 110)
    print("  CLIFF-ONLY EVALUATION: All Experiments on Activity Cliff Molecules")
    print("=" * 110)

    data_dir = get_data_dir()
    datasets = sorted([f for f in os.listdir(data_dir) if f.endswith(".csv")])
    all_res = []

    for i, d in enumerate(datasets):
        print(f"\n[{i+1}/{len(datasets)}] {d}...", end=" ", flush=True)
        t0 = time.time()
        try:
            r = run_dataset(d)
            if r is None: print("SKIP (too few cliffs)"); continue
            all_res.append(r); m = r["metrics"]
            print(f"({time.time()-t0:.1f}s) n_cliff={r['n_cliff_test']} "
                  f"SVM={m['SVM']['cliff_rmse']:.3f} "
                  f"RF+D={m['RF+Delta']['cliff_rmse']:.3f} "
                  f"GIN+D={m['GIN+Delta']['cliff_rmse']:.3f}")
        except Exception as e:
            print(f"ERROR: {e}"); import traceback; traceback.print_exc()

    # ============================================================
    # Table 1: Universal Delta on Cliff Molecules
    # ============================================================
    pairs = [("RF", "RF+Delta"), ("SVM", "SVM+Delta")]
    if HAS_XGB: pairs.append(("XGB", "XGB+Delta"))
    pairs += [("MLP", "MLP+Delta"), ("GIN", "GIN+Delta")]

    print("\n\n" + "=" * 110)
    print("  TABLE 1: BASE vs +DELTA on CLIFF MOLECULES ONLY")
    print("=" * 110)
    print(f"\n{'Model':10s} | {'Base Cliff':>10s} | {'+Delta Cliff':>13s} | {'Change':>8s} | {'Win':>5s}")
    print("-" * 60)

    for base, delta in pairs:
        b = [r["metrics"][base]["cliff_rmse"] for r in all_res]
        d = [r["metrics"][delta]["cliff_rmse"] for r in all_res]
        imp = np.mean([(bb-dd)/bb*100 for bb,dd in zip(b,d)])
        wins = sum(1 for bb,dd in zip(b,d) if dd < bb)
        print(f"{base:10s} | {np.mean(b):10.4f} | {np.mean(d):13.4f} | {imp:+7.2f}% | {wins}/{len(all_res)}")

    # Also show SVM+NeuralCorr and NB_Copy
    print(f"\n{'Extra models':30s} | {'Cliff RMSE':>10s}")
    print("-" * 45)
    for k in ["NB_Copy", "SVM+NeuralCorr"]:
        rmses = [r["metrics"][k]["cliff_rmse"] for r in all_res]
        print(f"{k:30s} | {np.mean(rmses):10.4f}")

    # ============================================================
    # Table 2: All models ranked on Cliff RMSE
    # ============================================================
    all_keys = ["SVM", "SVM+Delta", "SVM+NeuralCorr", "RF", "RF+Delta"]
    if HAS_XGB: all_keys += ["XGB", "XGB+Delta"]
    all_keys += ["MLP", "MLP+Delta", "GIN", "GIN+Delta", "NB_Copy"]

    print("\n\n" + "=" * 110)
    print("  TABLE 2: ALL MODELS RANKED by CLIFF RMSE")
    print("=" * 110)

    model_means = []
    for k in all_keys:
        rmses = [r["metrics"][k]["cliff_rmse"] for r in all_res]
        model_means.append((k, np.mean(rmses)))
    model_means.sort(key=lambda x: x[1])

    print(f"\n{'Rank':>4s} | {'Model':30s} | {'Cliff RMSE':>10s} | {'Win':>5s}")
    print("-" * 60)
    for rank, (k, mean_rmse) in enumerate(model_means, 1):
        wins = sum(1 for r in all_res
                   if k == min(all_keys, key=lambda kk: r["metrics"][kk]["cliff_rmse"]))
        print(f"{rank:4d} | {k:30s} | {mean_rmse:10.4f} | {wins}/{len(all_res)}")

    # ============================================================
    # Table 3: Per-dataset detail
    # ============================================================
    show_keys = ["SVM", "RF+Delta", "GIN+Delta", "NB_Copy"]
    short = {"SVM":"SVM", "RF+Delta":"RF+D", "GIN+Delta":"GIN+D", "NB_Copy":"NB"}

    print(f"\n\n{'':28s} | {'n_cl':>4s}", end="")
    for k in show_keys: print(f" | {short[k]:>5s}", end="")
    print(); print("-" * 70)

    for r in all_res:
        m = r["metrics"]
        vals = [m[k]["cliff_rmse"] for k in show_keys]
        bv = min(vals)
        line = f"{r['dataset']:28s} | {r['n_cliff_test']:4d}"
        for v in vals:
            mk = "*" if abs(v-bv)<1e-6 else " "
            line += f" | {v:4.3f}{mk}"
        print(line)

    means = [np.mean([r["metrics"][k]["cliff_rmse"] for r in all_res]) for k in show_keys]
    bm = min(means)
    line = f"{'MEAN':28s} | {'':>4s}"
    for v in means:
        mk = "*" if abs(v-bm)<1e-6 else " "
        line += f" | {v:4.3f}{mk}"
    print("-" * 70); print(line)

    # ============================================================
    # Table 4: Residual Analysis on Cliff Molecules
    # ============================================================
    print("\n\n" + "=" * 110)
    print("  TABLE 4: RESIDUAL ANALYSIS (CLIFF MOLECULES ONLY)")
    print("  'Can neighbor info explain model errors ON CLIFF COMPOUNDS?'")
    print("=" * 110)

    print(f"\n{'Model':8s} | {'Cliff RMSE':>10s} | {'Resid Std':>9s} | "
          f"{'Corr(res,Y_A)':>13s} | {'Corr(res,NB_d)':>14s}")
    print("-" * 70)

    for mname in ["SVM", "RF", "MLP", "GIN"]:
        rmses = [r["metrics"][mname]["cliff_rmse"] for r in all_res]
        stds = [r["analysis"][f"{mname}_resid_std"] for r in all_res]
        corr_ya = [r["analysis"][f"{mname}_corr_ya"] for r in all_res]
        corr_nb = [r["analysis"][f"{mname}_corr_nb"] for r in all_res]
        print(f"{mname:8s} | {np.mean(rmses):10.4f} | {np.mean(stds):9.4f} | "
              f"{np.mean(corr_ya):13.4f} | {np.mean(corr_nb):14.4f}")

    print("""
    Key question: On CLIFF molecules specifically,
    does neighbor info correlate more with model errors?
    If yes -> delta should help more on cliffs
    If no  -> cliff errors are fundamentally unpredictable from neighbors
    """)

    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "=" * 110)
    print("  SUMMARY (CLIFF MOLECULES ONLY)")
    print("=" * 110)
    for base, delta in pairs:
        b = [r["metrics"][base]["cliff_rmse"] for r in all_res]
        d = [r["metrics"][delta]["cliff_rmse"] for r in all_res]
        imp = np.mean([(bb-dd)/bb*100 for bb,dd in zip(b,d)])
        wins = sum(1 for bb,dd in zip(b,d) if dd < bb)
        print(f"  {base:8s} -> +Delta: {imp:+.2f}% cliff improvement, "
              f"wins {wins}/{len(all_res)} datasets")


if __name__ == "__main__":
    main()

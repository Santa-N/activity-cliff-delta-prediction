"""
GNN-based Ablation + Neighbor Strategy on MoleculeACE
=====================================================
All models use GIN encoder as backbone.
Part 1: Feature ablation (what inputs matter for correction?)
Part 2: Neighbor selection strategy comparison
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GINConv, global_add_pool
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, mean_absolute_error
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import RDLogger
import warnings
import os
import importlib.util
import time

warnings.filterwarnings("ignore")
RDLogger.logger().setLevel(RDLogger.ERROR)

np.random.seed(42)
torch.manual_seed(42)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FP_DIM = 1024
EMB_DIM = 128
NUM_GNN_LAYERS = 3

print(f"Device: {DEVICE}")


# ============================================================
# Mol -> Graph + FP
# ============================================================
ATOM_FEATURES = {
    'atom_type': list(range(1, 54)),
    'degree': [0, 1, 2, 3, 4, 5],
    'formal_charge': [-2, -1, 0, 1, 2],
    'hybridization': [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
    ],
    'aromatic': [False, True],
    'num_hs': [0, 1, 2, 3, 4],
}


def one_hot(val, allowed):
    enc = [0] * (len(allowed) + 1)
    if val in allowed:
        enc[allowed.index(val)] = 1
    else:
        enc[-1] = 1
    return enc


def atom_feats(atom):
    f = []
    f += one_hot(atom.GetAtomicNum(), ATOM_FEATURES['atom_type'])
    f += one_hot(atom.GetDegree(), ATOM_FEATURES['degree'])
    f += one_hot(atom.GetFormalCharge(), ATOM_FEATURES['formal_charge'])
    f += one_hot(atom.GetHybridization(), ATOM_FEATURES['hybridization'])
    f += one_hot(atom.GetIsAromatic(), ATOM_FEATURES['aromatic'])
    f += one_hot(atom.GetTotalNumHs(), ATOM_FEATURES['num_hs'])
    return f


ATOM_FEAT_DIM = sum(len(v) + 1 for v in ATOM_FEATURES.values())


def mol_to_graph(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    x = torch.FloatTensor([atom_feats(a) for a in mol.GetAtoms()])
    edges = []
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        edges += [[i, j], [j, i]]
    ei = torch.LongTensor(edges).t().contiguous() if edges else torch.zeros((2, 0), dtype=torch.long)
    return Data(x=x, edge_index=ei)


def smiles_to_fp(smi, nbits=FP_DIM):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=nbits)
    arr = np.zeros(nbits, dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def get_scaffold(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    except Exception:
        return None


# ============================================================
# Data
# ============================================================
def get_data_dir():
    spec = importlib.util.find_spec("MoleculeACE")
    return os.path.join(os.path.dirname(spec.origin), "Data", "benchmark_data")


def prepare_data(name):
    df = pd.read_csv(os.path.join(get_data_dir(), name))
    graphs, fps, ys, cliffs, splits, scaffolds = [], [], [], [], [], []
    for _, row in df.iterrows():
        g = mol_to_graph(row["smiles"])
        fp = smiles_to_fp(row["smiles"])
        if g is not None and fp is not None:
            graphs.append(g)
            fps.append(fp)
            ys.append(row["y"])
            cliffs.append(int(row["cliff_mol"]))
            splits.append(row["split"])
            scaffolds.append(get_scaffold(row["smiles"]))

    fps = np.array(fps)
    ys = np.array(ys, dtype=np.float32)
    cliffs = np.array(cliffs, dtype=bool)
    splits = np.array(splits)
    scaffolds = np.array(scaffolds, dtype=object)
    tr, te = splits == "train", splits == "test"

    return (
        [graphs[i] for i in range(len(graphs)) if tr[i]],
        fps[tr], ys[tr], cliffs[tr], scaffolds[tr],
        [graphs[i] for i in range(len(graphs)) if te[i]],
        fps[te], ys[te], cliffs[te], scaffolds[te],
    )


# ============================================================
# Neighbor Selection (Tanimoto-based, returns indices)
# ============================================================
def tanimoto_sim(a, b):
    a_ = a.astype(bool).astype(float)
    b_ = b.astype(bool).astype(float)
    inter = a_ @ b_.T
    return inter / np.maximum(a_.sum(1, keepdims=True) + b_.sum(1, keepdims=True).T - inter, 1e-8)


def select_tanimoto_top3(fps_q, fps_db, y_db, scaff_q, scaff_db, exclude_self=False):
    sim = tanimoto_sim(fps_q, fps_db)
    if exclude_self:
        np.fill_diagonal(sim, -1)
    n = len(fps_q)
    indices = np.zeros((n, 3), dtype=int)
    sims = np.zeros((n, 3), dtype=np.float32)
    for i in range(n):
        topk = np.argsort(-sim[i])[:3]
        indices[i] = topk
        sims[i] = sim[i][topk]
    return indices, sims


def select_scaffold_matched(fps_q, fps_db, y_db, scaff_q, scaff_db, exclude_self=False):
    sim = tanimoto_sim(fps_q, fps_db)
    if exclude_self:
        np.fill_diagonal(sim, -1)
    n = len(fps_q)
    indices = np.zeros((n, 3), dtype=int)
    sims = np.zeros((n, 3), dtype=np.float32)
    for i in range(n):
        if scaff_q[i] is not None:
            same = np.array([j for j in range(len(fps_db)) if scaff_db[j] == scaff_q[i]])
        else:
            same = np.array([], dtype=int)
        if len(same) >= 3:
            ss = sim[i][same]
            top3_local = same[np.argsort(-ss)[:3]]
            indices[i] = top3_local
            sims[i] = sim[i][top3_local]
        else:
            topk = np.argsort(-sim[i])[:3]
            indices[i] = topk
            sims[i] = sim[i][topk]
    return indices, sims


def select_cliff_potential(fps_q, fps_db, y_db, scaff_q, scaff_db, exclude_self=False):
    sim = tanimoto_sim(fps_q, fps_db)
    if exclude_self:
        np.fill_diagonal(sim, -1)
    n = len(fps_q)
    indices = np.zeros((n, 3), dtype=int)
    sims = np.zeros((n, 3), dtype=np.float32)
    for i in range(n):
        similar = np.where(sim[i] >= 0.4)[0]
        if len(similar) >= 3:
            mean_act = y_db[similar].mean()
            devs = np.abs(y_db[similar] - mean_act)
            top3_local = similar[np.argsort(-devs)[:3]]
            indices[i] = top3_local
            sims[i] = sim[i][top3_local]
        else:
            topk = np.argsort(-sim[i])[:3]
            indices[i] = topk
            sims[i] = sim[i][topk]
    return indices, sims


def select_random(fps_q, fps_db, y_db, scaff_q, scaff_db, exclude_self=False):
    sim = tanimoto_sim(fps_q, fps_db)
    n = len(fps_q)
    indices = np.zeros((n, 3), dtype=int)
    sims = np.zeros((n, 3), dtype=np.float32)
    for i in range(n):
        chosen = np.random.choice(len(fps_db), 3, replace=False)
        if exclude_self:
            while i in chosen:
                chosen = np.random.choice(len(fps_db), 3, replace=False)
        indices[i] = chosen
        sims[i] = sim[i][chosen]
    return indices, sims


def get_neighbor_embeddings(nb_indices, nb_sims, h_db, y_db):
    """Weighted average of top-k neighbor embeddings and activities."""
    n = len(nb_indices)
    h_nb = torch.zeros(n, h_db.shape[1])
    act_nb = np.zeros(n, dtype=np.float32)
    for i in range(n):
        w = nb_sims[i] / (nb_sims[i].sum() + 1e-8)
        wt = torch.FloatTensor(w)
        h_nb[i] = (h_db[nb_indices[i]] * wt.unsqueeze(-1)).sum(0)
        act_nb[i] = (y_db[nb_indices[i]] * w).sum()
    return h_nb, act_nb


# ============================================================
# GNN Models
# ============================================================
class GINEncoder(nn.Module):
    def __init__(self, in_dim, h_dim=EMB_DIM, n_layers=NUM_GNN_LAYERS, dropout=0.2):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for i in range(n_layers):
            d = in_dim if i == 0 else h_dim
            mlp = nn.Sequential(nn.Linear(d, h_dim), nn.ReLU(), nn.Linear(h_dim, h_dim))
            self.convs.append(GINConv(mlp))
            self.bns.append(nn.BatchNorm1d(h_dim))
        self.dropout = dropout

    def forward(self, data):
        x, ei, batch = data.x, data.edge_index, data.batch
        for conv, bn in zip(self.convs, self.bns):
            x = F.dropout(F.relu(bn(conv(x, ei))), self.dropout, self.training)
        return global_add_pool(x, batch)


class BaselineGNN(nn.Module):
    def __init__(self, in_dim, h_dim=EMB_DIM):
        super().__init__()
        self.encoder = GINEncoder(in_dim, h_dim)
        self.head = nn.Sequential(
            nn.Linear(h_dim, h_dim // 2), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(h_dim // 2, 1))

    def forward(self, data):
        return self.head(self.encoder(data)).squeeze(-1)

    def encode(self, data):
        return self.encoder(data)


class CorrectionNet(nn.Module):
    """Configurable correction on GNN embeddings."""
    def __init__(self, use_h_a=True, use_residual=True, use_y_a=True,
                 emb_dim=EMB_DIM):
        super().__init__()
        self.use_h_a = use_h_a
        self.use_residual = use_residual
        self.use_y_a = use_y_a

        in_dim = emb_dim  # h_B always
        if use_h_a:
            in_dim += emb_dim
        if use_residual:
            in_dim += emb_dim
        if use_y_a:
            in_dim += 1
        in_dim += 1  # m(B)

        self.net = nn.Sequential(
            nn.Linear(in_dim, emb_dim), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(emb_dim, emb_dim // 2), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(emb_dim // 2, 1))

    def forward(self, h_b, h_a, y_a, m_b):
        parts = [h_b]
        if self.use_h_a:
            parts.append(h_a)
        if self.use_residual:
            parts.append(h_b - h_a)
        if self.use_y_a:
            parts.append(y_a.unsqueeze(-1))
        parts.append(m_b.unsqueeze(-1))
        return self.net(torch.cat(parts, -1)).squeeze(-1)


# ============================================================
# Training
# ============================================================
def train_gnn_baseline(graphs_tr, y_tr, graphs_va, y_va,
                        epochs=150, lr=1e-3, patience=25, bs=64):
    model = BaselineGNN(ATOM_FEAT_DIM).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=10, factor=0.5, min_lr=1e-5)
    crit = nn.MSELoss()

    va_batch = Batch.from_data_list(graphs_va).to(DEVICE)
    va_y = torch.FloatTensor(y_va).to(DEVICE)

    best_vl, best_st, wait = float("inf"), None, 0
    n = len(graphs_tr)

    for _ in range(epochs):
        model.train()
        perm = np.random.permutation(n)
        for s in range(0, n, bs):
            idx = perm[s:s+bs]
            batch = Batch.from_data_list([graphs_tr[i] for i in idx]).to(DEVICE)
            yt = torch.FloatTensor(y_tr[idx]).to(DEVICE)
            loss = crit(model(batch), yt)
            opt.zero_grad(); loss.backward(); opt.step()

        model.eval()
        with torch.no_grad():
            vl = crit(model(va_batch), va_y).item()
        sched.step(vl)
        if vl < best_vl:
            best_vl = vl
            best_st = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    model.load_state_dict(best_st)
    return model


def get_embeddings(model, graphs, bs=256):
    model.eval()
    embs = []
    for s in range(0, len(graphs), bs):
        batch = Batch.from_data_list(graphs[s:s+bs]).to(DEVICE)
        with torch.no_grad():
            embs.append(model.encode(batch).cpu())
    return torch.cat(embs, 0)


def train_correction(corr, h_tr, y_tr, h_nb_tr, nba_tr, m_tr,
                      h_va, y_va, h_nb_va, nba_va, m_va,
                      epochs=200, lr=5e-4, patience=25, bs=64):
    opt = torch.optim.Adam(corr.parameters(), lr=lr, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=10, factor=0.5, min_lr=1e-5)

    ds = TensorDataset(h_tr, torch.FloatTensor(y_tr), h_nb_tr,
                        torch.FloatTensor(nba_tr), torch.FloatTensor(m_tr))
    loader = DataLoader(ds, batch_size=bs, shuffle=True)

    H_v, Y_v = h_va.to(DEVICE), torch.FloatTensor(y_va).to(DEVICE)
    HN_v = h_nb_va.to(DEVICE)
    NA_v, M_v = torch.FloatTensor(nba_va).to(DEVICE), torch.FloatTensor(m_va).to(DEVICE)

    best_vl, best_st, wait = float("inf"), None, 0
    for _ in range(epochs):
        corr.train()
        for hb, yb, hnb, nab, mb in loader:
            hb, yb, hnb, nab, mb = [t.to(DEVICE) for t in [hb, yb, hnb, nab, mb]]
            loss = nn.MSELoss()(mb + corr(hb, hnb, nab, mb), yb)
            opt.zero_grad(); loss.backward(); opt.step()
        corr.eval()
        with torch.no_grad():
            vl = nn.MSELoss()(M_v + corr(H_v, HN_v, NA_v, M_v), Y_v).item()
        sched.step(vl)
        if vl < best_vl:
            best_vl = vl
            best_st = {k: v.clone() for k, v in corr.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break
    corr.load_state_dict(best_st)
    return corr


# ============================================================
# Run one dataset
# ============================================================
def run_dataset(name):
    (graphs_tr, fps_tr, y_tr, cliff_tr, scaff_tr,
     graphs_te, fps_te, y_te, cliff_te, scaff_te) = prepare_data(name)

    if len(graphs_tr) < 20 or len(graphs_te) < 5:
        return None

    # Val split
    n = len(graphs_tr)
    perm = np.random.permutation(n)
    n_va = max(int(n * 0.15), 10)
    va_i, tr_i = perm[:n_va], perm[n_va:]

    g_va = [graphs_tr[i] for i in va_i]
    fps_va, y_va, scaff_va = fps_tr[va_i], y_tr[va_i], scaff_tr[va_i]
    g_tr = [graphs_tr[i] for i in tr_i]
    fps_tr2, y_tr2, scaff_tr2 = fps_tr[tr_i], y_tr[tr_i], scaff_tr[tr_i]

    # Train GNN baseline (shared encoder)
    base_model = train_gnn_baseline(g_tr, y_tr2, g_va, y_va)
    base_model.eval()

    # Get embeddings
    h_tr = get_embeddings(base_model, g_tr)
    h_va = get_embeddings(base_model, g_va)
    h_te = get_embeddings(base_model, graphs_te)
    h_all_tr = get_embeddings(base_model, graphs_tr)

    # Baseline predictions m(B)
    with torch.no_grad():
        m_tr = base_model(Batch.from_data_list(g_tr).to(DEVICE)).cpu().numpy()
        m_va = base_model(Batch.from_data_list(g_va).to(DEVICE)).cpu().numpy()
        te_batch = Batch.from_data_list(graphs_te).to(DEVICE)
        m_te = base_model(te_batch).cpu().numpy()

    results = {"Baseline": m_te.copy()}

    # ---- Neighbor strategies ----
    strategies = {
        "Tanimoto_Top3": select_tanimoto_top3,
        "Scaffold_Match": select_scaffold_matched,
        "Cliff_Potential": select_cliff_potential,
        "Random": select_random,
    }

    strat_data = {}
    for sname, sfunc in strategies.items():
        idx_tr, sim_tr = sfunc(fps_tr2, fps_tr2, y_tr2, scaff_tr2, scaff_tr2, exclude_self=True)
        idx_va, sim_va = sfunc(fps_va, fps_tr2, y_tr2, scaff_va, scaff_tr2)
        idx_te, sim_te = sfunc(fps_te, fps_tr, y_tr, scaff_te, scaff_tr)

        h_nb_tr, nba_tr = get_neighbor_embeddings(idx_tr, sim_tr, h_tr, y_tr2)
        h_nb_va, nba_va = get_neighbor_embeddings(idx_va, sim_va, h_tr, y_tr2)
        h_nb_te, nba_te = get_neighbor_embeddings(idx_te, sim_te, h_all_tr, y_tr)

        strat_data[sname] = (h_nb_tr, nba_tr, h_nb_va, nba_va, h_nb_te, nba_te)

    # ---- Part 1: Ablation (Tanimoto_Top3 strategy) ----
    sd = strat_data["Tanimoto_Top3"]
    h_nb_tr, nba_tr, h_nb_va, nba_va, h_nb_te, nba_te = sd

    ablation_configs = {
        "No_Neighbor":  {"use_h_a": False, "use_residual": False, "use_y_a": False},
        "Full":         {"use_h_a": True,  "use_residual": True,  "use_y_a": True},
        "No_Residual":  {"use_h_a": True,  "use_residual": False, "use_y_a": True},
        "No_H_A":       {"use_h_a": False, "use_residual": True,  "use_y_a": True},
        "No_Y_A":       {"use_h_a": True,  "use_residual": True,  "use_y_a": False},
        "Resid_Only":   {"use_h_a": False, "use_residual": True,  "use_y_a": False},
        "Y_A_Only":     {"use_h_a": False, "use_residual": False, "use_y_a": True},
    }

    for aname, cfg in ablation_configs.items():
        corr = CorrectionNet(**cfg).to(DEVICE)
        corr = train_correction(corr, h_tr, y_tr2, h_nb_tr, nba_tr, m_tr,
                                 h_va, y_va, h_nb_va, nba_va, m_va)
        corr.eval()
        with torch.no_grad():
            h_te_d, h_nb_te_d = h_te.to(DEVICE), h_nb_te.to(DEVICE)
            na_te_d = torch.FloatTensor(nba_te).to(DEVICE)
            m_te_d = torch.FloatTensor(m_te).to(DEVICE)
            pred = (m_te_d + corr(h_te_d, h_nb_te_d, na_te_d, m_te_d)).cpu().numpy()
        results[f"Abl_{aname}"] = pred

    # ---- Part 2: Strategy comparison (Full config) ----
    for sname in strategies:
        sd = strat_data[sname]
        h_nb_tr_s, nba_tr_s, h_nb_va_s, nba_va_s, h_nb_te_s, nba_te_s = sd

        corr = CorrectionNet(use_h_a=True, use_residual=True, use_y_a=True).to(DEVICE)
        corr = train_correction(corr, h_tr, y_tr2, h_nb_tr_s, nba_tr_s, m_tr,
                                 h_va, y_va, h_nb_va_s, nba_va_s, m_va)
        corr.eval()
        with torch.no_grad():
            h_te_d = h_te.to(DEVICE)
            h_nb_d = h_nb_te_s.to(DEVICE)
            na_d = torch.FloatTensor(nba_te_s).to(DEVICE)
            m_d = torch.FloatTensor(m_te).to(DEVICE)
            pred = (m_d + corr(h_te_d, h_nb_d, na_d, m_d)).cpu().numpy()
        results[f"Strat_{sname}"] = pred

    # ---- Metrics ----
    def calc(yt, yp):
        rmse = np.sqrt(mean_squared_error(yt, yp))
        r = np.corrcoef(yt, yp)[0, 1] if len(yt) > 1 else 0
        return rmse, r

    out = {"dataset": name, "n_test": len(fps_te),
           "n_cliff": int(cliff_te.sum()), "metrics": {}}
    for rn, pred in results.items():
        ov = calc(y_te, pred)
        cl = calc(y_te[cliff_te], pred[cliff_te]) if cliff_te.sum() > 1 else (np.nan, np.nan)
        out["metrics"][rn] = {"overall_rmse": ov[0], "overall_r": ov[1],
                              "cliff_rmse": cl[0], "cliff_r": cl[1]}
    return out


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 95)
    print("  GNN Ablation + Neighbor Strategy (MoleculeACE)")
    print("=" * 95)

    data_dir = get_data_dir()
    datasets = sorted([f for f in os.listdir(data_dir) if f.endswith(".csv")])

    all_res = []
    for i, d in enumerate(datasets):
        print(f"\n[{i+1}/{len(datasets)}] {d}...", end=" ", flush=True)
        t0 = time.time()
        try:
            r = run_dataset(d)
            if r is None:
                print("SKIP"); continue
            all_res.append(r)
            m = r["metrics"]
            print(f"({time.time()-t0:.1f}s) "
                  f"Base={m['Baseline']['overall_rmse']:.3f} "
                  f"NoNB={m['Abl_No_Neighbor']['overall_rmse']:.3f} "
                  f"Full={m['Abl_Full']['overall_rmse']:.3f}")
        except Exception as e:
            print(f"ERROR: {e}")
            import traceback; traceback.print_exc()

    # ============================================================
    # Results
    # ============================================================
    abl_keys = ["Baseline", "Abl_No_Neighbor", "Abl_Full", "Abl_No_Residual",
                "Abl_No_H_A", "Abl_No_Y_A", "Abl_Resid_Only", "Abl_Y_A_Only"]
    abl_labels = {
        "Baseline":         "GNN Baseline m(B)",
        "Abl_No_Neighbor":  "m(B)+g(h_B, m(B)) no neighbor",
        "Abl_Full":         "Full: h_B+h_A+Resid+Y(A)",
        "Abl_No_Residual":  "- Residual: h_B+h_A+Y(A)",
        "Abl_No_H_A":       "- h(A): h_B+Resid+Y(A)",
        "Abl_No_Y_A":       "- Y(A): h_B+h_A+Resid",
        "Abl_Resid_Only":   "Resid only: h_B+Resid",
        "Abl_Y_A_Only":     "Y(A) only: h_B+Y(A)",
    }

    print("\n\n" + "=" * 95)
    print("  PART 1: GNN FEATURE ABLATION (Tanimoto Top-3)")
    print("=" * 95)
    print(f"\n{'Config':42s} | {'Overall RMSE':>12s} | {'Cliff RMSE':>10s} | {'R':>6s} | {'vs Base':>8s} | {'Win':>5s}")
    print("-" * 95)

    base_rmses = [r["metrics"]["Baseline"]["overall_rmse"] for r in all_res]
    for key in abl_keys:
        rmses = [r["metrics"][key]["overall_rmse"] for r in all_res]
        cliffs = [r["metrics"][key]["cliff_rmse"] for r in all_res
                  if not np.isnan(r["metrics"][key]["cliff_rmse"])]
        rs = [r["metrics"][key]["overall_r"] for r in all_res]
        if key == "Baseline":
            imp, wins = "---", "---"
        else:
            diffs = [(b-v)/b*100 for b, v in zip(base_rmses, rmses)]
            imp = f"{np.mean(diffs):+.2f}%"
            wins = f"{sum(1 for d in diffs if d > 0)}/{len(diffs)}"
        print(f"{abl_labels[key]:42s} | {np.mean(rmses):12.4f} | "
              f"{np.mean(cliffs):10.4f} | {np.mean(rs):6.4f} | {imp:>8s} | {wins:>5s}")

    # Per-dataset
    short = ["Baseline", "Abl_No_Neighbor", "Abl_Full", "Abl_No_Residual",
             "Abl_No_H_A", "Abl_No_Y_A", "Abl_Resid_Only", "Abl_Y_A_Only"]
    headers = ["Base", "NoNB", "Full", "NoRes", "NoHA", "NoYA", "ResO", "YAO"]

    print(f"\n{'':28s}", end="")
    for h in headers:
        print(f" | {h:>6s}", end="")
    print()
    print("-" * 95)

    for r in all_res:
        m = r["metrics"]
        vals = [m[k]["overall_rmse"] for k in short]
        bv = min(vals)
        line = f"{r['dataset']:28s}"
        for v in vals:
            mk = "*" if v == bv else " "
            line += f" | {v:5.3f}{mk}"
        print(line)

    means = [np.mean([r["metrics"][k]["overall_rmse"] for r in all_res]) for k in short]
    bm = min(means)
    line = f"{'MEAN':28s}"
    for v in means:
        mk = "*" if v == bm else " "
        line += f" | {v:5.3f}{mk}"
    print("-" * 95)
    print(line)

    # ============================================================
    # Part 2: Strategy
    # ============================================================
    strat_keys = ["Strat_Tanimoto_Top3", "Strat_Scaffold_Match",
                  "Strat_Cliff_Potential", "Strat_Random"]
    strat_labels = {
        "Strat_Tanimoto_Top3": "Tanimoto Top-3",
        "Strat_Scaffold_Match": "Scaffold-matched",
        "Strat_Cliff_Potential": "Cliff-potential",
        "Strat_Random": "Random (control)",
    }

    print("\n\n" + "=" * 95)
    print("  PART 2: GNN NEIGHBOR STRATEGY (Full correction)")
    print("=" * 95)

    print(f"\n{'Strategy':35s} | {'Overall RMSE':>12s} | {'Cliff RMSE':>10s} | {'vs Base':>8s} | {'Win':>5s}")
    print("-" * 80)

    print(f"{'GNN Baseline':35s} | {np.mean(base_rmses):12.4f} | "
          f"{np.mean([r['metrics']['Baseline']['cliff_rmse'] for r in all_res if not np.isnan(r['metrics']['Baseline']['cliff_rmse'])]):10.4f} | "
          f"{'---':>8s} | {'---':>5s}")

    for key in strat_keys:
        rmses = [r["metrics"][key]["overall_rmse"] for r in all_res]
        cliffs = [r["metrics"][key]["cliff_rmse"] for r in all_res
                  if not np.isnan(r["metrics"][key]["cliff_rmse"])]
        diffs = [(b-v)/b*100 for b, v in zip(base_rmses, rmses)]
        wins = sum(1 for d in diffs if d > 0)
        print(f"{strat_labels[key]:35s} | {np.mean(rmses):12.4f} | "
              f"{np.mean(cliffs):10.4f} | {np.mean(diffs):+7.2f}% | {wins}/{len(diffs)}")

    # Per-dataset strategy
    all_sk = ["Baseline"] + strat_keys
    sheaders = ["Base", "Top3", "Scaff", "Cliff", "Rand"]
    print(f"\n{'':28s}", end="")
    for h in sheaders:
        print(f" | {h:>6s}", end="")
    print()
    print("-" * 75)

    for r in all_res:
        m = r["metrics"]
        vals = [m[k]["overall_rmse"] for k in all_sk]
        bv = min(vals)
        line = f"{r['dataset']:28s}"
        for v in vals:
            mk = "*" if v == bv else " "
            line += f" | {v:5.3f}{mk}"
        print(line)

    means = [np.mean([r["metrics"][k]["overall_rmse"] for r in all_res]) for k in all_sk]
    bm = min(means)
    line = f"{'MEAN':28s}"
    for v in means:
        mk = "*" if v == bm else " "
        line += f" | {v:5.3f}{mk}"
    print("-" * 75)
    print(line)

    # Cliff subset
    print(f"\n[Cliff Subset RMSE]")
    print(f"{'':28s}", end="")
    for h in sheaders:
        print(f" | {h:>6s}", end="")
    print()
    print("-" * 75)
    for r in all_res:
        m = r["metrics"]
        vals = [m[k]["cliff_rmse"] for k in all_sk]
        valid = [v for v in vals if not np.isnan(v)]
        if not valid:
            continue
        bv = min(valid)
        line = f"{r['dataset']:28s}"
        for v in vals:
            if np.isnan(v):
                line += f" |   nan "
            else:
                mk = "*" if v == bv else " "
                line += f" | {v:5.3f}{mk}"
        print(line)

    cmeans = []
    for k in all_sk:
        cm = [r["metrics"][k]["cliff_rmse"] for r in all_res
              if not np.isnan(r["metrics"][k]["cliff_rmse"])]
        cmeans.append(np.mean(cm))
    bcm = min(cmeans)
    line = f"{'MEAN':28s}"
    for v in cmeans:
        mk = "*" if v == bcm else " "
        line += f" | {v:5.3f}{mk}"
    print("-" * 75)
    print(line)


if __name__ == "__main__":
    main()

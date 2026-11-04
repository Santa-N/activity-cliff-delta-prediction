"""
Structural Transformation Rule Learning
========================================
Instead of generic h_B - h_A, explicitly model:
  - ADDED: substructures in B but not A
  - REMOVED: substructures in A but not B
  - CONTEXT: shared scaffold/substructures

The transformation rule network learns:
  "Adding substructure X in context C -> activity change delta"

This captures MMP-like transformation rules in a data-driven way.

Models compared:
  1. SVM (reference)
  2. GIN Baseline
  3. GIN + Delta (previous: Y_A + MLP(h_B, h_A, h_B-h_A))
  4. FP + Transform Rule (added/removed/context from fingerprints)
  5. GIN + Transform Rule (added/removed/context from embeddings)
  6. GIN + Gated Transform (learnable gate for activity-relevant changes)
  7. SVM + Transform Rule (SVM predicts delta from transform features)
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GINConv, global_add_pool
from torch.utils.data import DataLoader, TensorDataset
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit import RDLogger
import warnings, os, importlib.util, time

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
# Data + Neighbors
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
# Compute structural transformation features (FP-based)
# ============================================================
def compute_transform_features_fp(fp_b, fp_a):
    """
    From fingerprints, compute:
      added:   bits in B but not in A (new substructures)
      removed: bits in A but not in B (lost substructures)
      context: bits shared between A and B (scaffold context)
    """
    # For binary FP: treat > 0.5 as "on"
    b_bin = (fp_b > 0.5).astype(np.float32)
    a_bin = (fp_a > 0.5).astype(np.float32)

    added = b_bin * (1 - a_bin)      # in B, not in A
    removed = a_bin * (1 - b_bin)    # in A, not in B
    context = b_bin * a_bin           # shared

    return added, removed, context


# ============================================================
# Models
# ============================================================

# --- GIN Encoder ---
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


# --- Model 3: GIN + simple Delta (baseline) ---
class GINDelta(nn.Module):
    def __init__(self, emb=EMB_DIM):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(emb*3,emb),nn.ReLU(),nn.Dropout(0.3),
                                  nn.Linear(emb,emb//2),nn.ReLU(),nn.Dropout(0.3),nn.Linear(emb//2,1))
    def forward(self, h_b, h_a, y_a):
        return y_a + self.net(torch.cat([h_b, h_a, h_b-h_a],-1)).squeeze(-1)


# --- Model 4: FP Transform Rule ---
class FPTransformRule(nn.Module):
    """
    Learn transformation rules from fingerprint-level structural changes.
    added/removed/context -> activity change
    """
    def __init__(self, dim=FP_DIM, h=256):
        super().__init__()
        # Encode each transformation component separately
        self.added_enc = nn.Sequential(nn.Linear(dim, h//2), nn.ReLU())
        self.removed_enc = nn.Sequential(nn.Linear(dim, h//2), nn.ReLU())
        self.context_enc = nn.Sequential(nn.Linear(dim, h//2), nn.ReLU())

        # Combine: context modulates the effect of added/removed
        self.rule_net = nn.Sequential(
            nn.Linear(h//2 * 3, h), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(h, h//2), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(h//2, 1))

    def forward(self, added, removed, context, y_a):
        h_add = self.added_enc(added)
        h_rem = self.removed_enc(removed)
        h_ctx = self.context_enc(context)
        delta = self.rule_net(torch.cat([h_add, h_rem, h_ctx], -1)).squeeze(-1)
        return y_a + delta


# --- Model 5: GIN Transform Rule ---
class GINTransformRule(nn.Module):
    """
    Transform rules in GNN embedding space.
    added = ReLU(h_B - h_A)   what B gained
    removed = ReLU(h_A - h_B) what B lost
    context = h_B * h_A       shared representation
    """
    def __init__(self, emb=EMB_DIM):
        super().__init__()
        self.added_enc = nn.Sequential(nn.Linear(emb, emb), nn.ReLU())
        self.removed_enc = nn.Sequential(nn.Linear(emb, emb), nn.ReLU())
        self.context_enc = nn.Sequential(nn.Linear(emb, emb), nn.ReLU())

        self.rule_net = nn.Sequential(
            nn.Linear(emb*3, emb), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(emb, emb//2), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(emb//2, 1))

    def forward(self, h_b, h_a, y_a):
        added = F.relu(h_b - h_a)     # what B gained in embedding space
        removed = F.relu(h_a - h_b)   # what B lost
        context = h_b * h_a           # shared scaffold encoding

        h_add = self.added_enc(added)
        h_rem = self.removed_enc(removed)
        h_ctx = self.context_enc(context)

        delta = self.rule_net(torch.cat([h_add, h_rem, h_ctx], -1)).squeeze(-1)
        return y_a + delta


# --- Model 6: GIN Gated Transform ---
class GINGatedTransform(nn.Module):
    """
    Gated transformation: learn which structural changes are activity-relevant.
    Gate = sigmoid(W * [h_B, h_A]) -> soft mask on which dimensions changed meaningfully
    Then apply gated diff to predict delta.
    """
    def __init__(self, emb=EMB_DIM):
        super().__init__()
        # Gate: which dimensions of the change are activity-relevant?
        self.gate = nn.Sequential(
            nn.Linear(emb*2, emb), nn.Sigmoid())

        # Transform rule on gated changes
        self.added_enc = nn.Sequential(nn.Linear(emb, emb), nn.ReLU())
        self.removed_enc = nn.Sequential(nn.Linear(emb, emb), nn.ReLU())

        self.rule_net = nn.Sequential(
            nn.Linear(emb*3, emb), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(emb, emb//2), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(emb//2, 1))

    def forward(self, h_b, h_a, y_a):
        # Learn which changes matter for activity
        relevance = self.gate(torch.cat([h_b, h_a], -1))  # (batch, emb)

        diff = h_b - h_a
        gated_diff = diff * relevance  # only activity-relevant changes

        added = F.relu(gated_diff)
        removed = F.relu(-gated_diff)
        context = h_b * h_a

        h_add = self.added_enc(added)
        h_rem = self.removed_enc(removed)

        delta = self.rule_net(torch.cat([h_add, h_rem, context], -1)).squeeze(-1)
        return y_a + delta


# ============================================================
# Training
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

def train_delta_model(net, tensors_tr, tensors_va, predict_fn,
                       epochs=200, lr=5e-4, pat=25, bs=64):
    """Generic trainer for delta-style models."""
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=10, factor=0.5, min_lr=1e-5)
    ds = TensorDataset(*tensors_tr)
    loader = DataLoader(ds, batch_size=bs, shuffle=True)
    va_tensors = [t.to(DEVICE) for t in tensors_va]
    best, bst, w = float("inf"), None, 0
    for _ in range(epochs):
        net.train()
        for batch in loader:
            batch = [t.to(DEVICE) for t in batch]
            pred = predict_fn(net, batch)
            loss = nn.MSELoss()(pred, batch[1])
            opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            vl = nn.MSELoss()(predict_fn(net, va_tensors), va_tensors[1]).item()
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

    t = lambda x: torch.FloatTensor(x)
    results = {}

    # FP neighbors
    nb_fps_tr, nb_y_tr = get_neighbors(fps_tr, fps_tr, y_tr, exclude_self=True)
    nb_fps_te, nb_y_te = get_neighbors(fps_te, fps_tr, y_tr)
    nb_fps_tr2, nb_y_tr2 = get_neighbors(fps_tr2, fps_tr2, y_tr2, exclude_self=True)
    nb_fps_va, nb_y_va = get_neighbors(fps_va, fps_tr2, y_tr2)

    # --- 1. SVM ---
    svm = SVR(kernel='rbf', C=10, gamma='scale')
    svm.fit(fps_tr, y_tr)
    results["SVM"] = svm.predict(fps_te)

    # --- 2. GIN Baseline ---
    gin = train_gin(g_tr2, y_tr2, g_va, y_va)
    gin.eval()
    with torch.no_grad():
        results["GIN"] = gin(Batch.from_data_list(g_te).to(DEVICE)).cpu().numpy()

    # GNN embeddings
    h_tr = get_embs(gin, g_tr2); h_va = get_embs(gin, g_va)
    h_te = get_embs(gin, g_te); h_all = get_embs(gin, g_tr)
    h_nb_tr, y_nb_tr = get_emb_neighbors(fps_tr2, fps_tr2, h_tr, y_tr2, exclude_self=True)
    h_nb_va, y_nb_va = get_emb_neighbors(fps_va, fps_tr2, h_tr, y_tr2)
    h_nb_te, y_nb_te = get_emb_neighbors(fps_te, fps_tr, h_all, y_tr)

    # --- 3. GIN + Delta (previous best) ---
    gd = GINDelta().to(DEVICE)
    pred_fn_delta = lambda net, b: net(b[0], b[2], b[3])
    tensors_tr = [t(h_tr), t(y_tr2), t(h_nb_tr), t(y_nb_tr)]
    tensors_va = [t(h_va), t(y_va), t(h_nb_va), t(y_nb_va)]
    gd = train_delta_model(gd, tensors_tr, tensors_va, pred_fn_delta)
    gd.eval()
    with torch.no_grad():
        results["GIN+Delta"] = gd(h_te.to(DEVICE), h_nb_te.to(DEVICE),
                                   t(y_nb_te).to(DEVICE)).cpu().numpy()

    # --- 4. FP + Transform Rule ---
    add_tr, rem_tr, ctx_tr = compute_transform_features_fp(fps_tr2, nb_fps_tr2)
    add_va, rem_va, ctx_va = compute_transform_features_fp(fps_va, nb_fps_va)
    add_te, rem_te, ctx_te = compute_transform_features_fp(fps_te, nb_fps_te)

    fp_tr_net = FPTransformRule().to(DEVICE)
    pred_fn_fp_tr = lambda net, b: net(b[2], b[3], b[4], b[5])
    tensors_tr_fp = [t(fps_tr2), t(y_tr2), t(add_tr), t(rem_tr), t(ctx_tr), t(nb_y_tr2)]
    tensors_va_fp = [t(fps_va), t(y_va), t(add_va), t(rem_va), t(ctx_va), t(nb_y_va)]
    fp_tr_net = train_delta_model(fp_tr_net, tensors_tr_fp, tensors_va_fp, pred_fn_fp_tr)
    fp_tr_net.eval()
    with torch.no_grad():
        results["FP+TransformRule"] = fp_tr_net(
            t(add_te).to(DEVICE), t(rem_te).to(DEVICE),
            t(ctx_te).to(DEVICE), t(nb_y_te).to(DEVICE)).cpu().numpy()

    # --- 5. GIN + Transform Rule ---
    gin_tr = GINTransformRule().to(DEVICE)
    pred_fn_gin_tr = lambda net, b: net(b[0], b[2], b[3])
    gin_tr = train_delta_model(gin_tr, tensors_tr, tensors_va, pred_fn_gin_tr)
    gin_tr.eval()
    with torch.no_grad():
        results["GIN+TransformRule"] = gin_tr(
            h_te.to(DEVICE), h_nb_te.to(DEVICE),
            t(y_nb_te).to(DEVICE)).cpu().numpy()

    # --- 6. GIN + Gated Transform ---
    gin_gt = GINGatedTransform().to(DEVICE)
    pred_fn_gt = lambda net, b: net(b[0], b[2], b[3])
    gin_gt = train_delta_model(gin_gt, tensors_tr, tensors_va, pred_fn_gt)
    gin_gt.eval()
    with torch.no_grad():
        results["GIN+GatedTransform"] = gin_gt(
            h_te.to(DEVICE), h_nb_te.to(DEVICE),
            t(y_nb_te).to(DEVICE)).cpu().numpy()

    # --- 7. SVM + Transform Rule ---
    delta_y_tr = y_tr - nb_y_tr
    add_tr_full, rem_tr_full, ctx_tr_full = compute_transform_features_fp(fps_tr, nb_fps_tr)
    X_tr_svm = np.concatenate([add_tr_full, rem_tr_full, ctx_tr_full], axis=1)
    svm_tr = SVR(kernel='rbf', C=10, gamma='scale')
    svm_tr.fit(X_tr_svm, delta_y_tr)
    add_te_full, rem_te_full, ctx_te_full = compute_transform_features_fp(fps_te, nb_fps_te)
    X_te_svm = np.concatenate([add_te_full, rem_te_full, ctx_te_full], axis=1)
    results["SVM+TransformRule"] = nb_y_te + svm_tr.predict(X_te_svm)

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
    print("  Structural Transformation Rule Learning")
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
                  f"SVM={m['SVM']['overall_rmse']:.3f} "
                  f"GIN={m['GIN']['overall_rmse']:.3f} "
                  f"Delta={m['GIN+Delta']['overall_rmse']:.3f} "
                  f"FP_TR={m['FP+TransformRule']['overall_rmse']:.3f} "
                  f"GIN_TR={m['GIN+TransformRule']['overall_rmse']:.3f} "
                  f"Gated={m['GIN+GatedTransform']['overall_rmse']:.3f} "
                  f"SVM_TR={m['SVM+TransformRule']['overall_rmse']:.3f}")
        except Exception as e:
            print(f"ERROR: {e}"); import traceback; traceback.print_exc()

    # Results
    keys = ["SVM", "GIN", "GIN+Delta", "FP+TransformRule",
            "GIN+TransformRule", "GIN+GatedTransform", "SVM+TransformRule"]
    short = {"SVM":"SVM", "GIN":"GIN", "GIN+Delta":"G+D",
             "FP+TransformRule":"FP+TR", "GIN+TransformRule":"G+TR",
             "GIN+GatedTransform":"G+GT", "SVM+TransformRule":"S+TR"}

    print("\n\n" + "=" * 110)
    print("  RESULTS")
    print("=" * 110)
    print(f"\n{'Model':25s} | {'RMSE':>8s} | {'Cliff':>8s} | {'Win_OV':>6s} | {'Win_CL':>6s}")
    print("-" * 65)

    for key in keys:
        rmses = [r["metrics"][key]["overall_rmse"] for r in all_res]
        cliffs = [r["metrics"][key]["cliff_rmse"] for r in all_res
                  if not np.isnan(r["metrics"][key]["cliff_rmse"])]
        ov_w = sum(1 for r in all_res
                   if key == min(keys, key=lambda k: r["metrics"][k]["overall_rmse"]))
        cl_w = sum(1 for r in all_res
                   if not np.isnan(r["metrics"][key]["cliff_rmse"]) and
                   key == min([k for k in keys if not np.isnan(r["metrics"][k]["cliff_rmse"])],
                              key=lambda k: r["metrics"][k]["cliff_rmse"]))
        print(f"{key:25s} | {np.mean(rmses):8.4f} | {np.mean(cliffs):8.4f} | {ov_w:>6d} | {cl_w:>6d}")

    # Per-dataset
    print(f"\n{'':28s}", end="")
    for k in keys: print(f" | {short[k]:>5s}", end="")
    print(); print("-" * 85)

    for r in all_res:
        m = r["metrics"]
        vals = [m[k]["overall_rmse"] for k in keys]
        bv = min(vals)
        line = f"{r['dataset']:28s}"
        for v in vals:
            mk = "*" if abs(v-bv)<1e-6 else " "
            line += f" | {v:4.3f}{mk}"
        print(line)

    means = [np.mean([r["metrics"][k]["overall_rmse"] for r in all_res]) for k in keys]
    bm = min(means)
    line = f"{'MEAN':28s}"
    for v in means:
        mk = "*" if abs(v-bm)<1e-6 else " "
        line += f" | {v:4.3f}{mk}"
    print("-" * 85); print(line)

    # Cliff
    print(f"\n[Cliff RMSE]")
    print(f"{'':28s}", end="")
    for k in keys: print(f" | {short[k]:>5s}", end="")
    print(); print("-" * 85)

    for r in all_res:
        m = r["metrics"]
        vals = [m[k]["cliff_rmse"] for k in keys]
        valid = [v for v in vals if not np.isnan(v)]
        if not valid: continue
        bv = min(valid)
        line = f"{r['dataset']:28s}"
        for v in vals:
            if np.isnan(v): line += f" |  nan "
            else:
                mk = "*" if abs(v-bv)<1e-6 else " "
                line += f" | {v:4.3f}{mk}"
        print(line)

    cmeans = [np.mean([r["metrics"][k]["cliff_rmse"] for r in all_res
                       if not np.isnan(r["metrics"][k]["cliff_rmse"])]) for k in keys]
    bcm = min(cmeans)
    line = f"{'MEAN':28s}"
    for v in cmeans:
        mk = "*" if abs(v-bcm)<1e-6 else " "
        line += f" | {v:4.3f}{mk}"
    print("-" * 85); print(line)

    # Delta vs Transform comparison
    print("\n\n" + "=" * 110)
    print("  DELTA vs TRANSFORM RULE (same backbone)")
    print("=" * 110)
    comparisons = [
        ("GIN+Delta", "GIN+TransformRule", "GIN+GatedTransform"),
        ("SVM", "SVM+TransformRule", None),
    ]
    for base, tr, gated in comparisons:
        b_rmses = [r["metrics"][base]["overall_rmse"] for r in all_res]
        t_rmses = [r["metrics"][tr]["overall_rmse"] for r in all_res]
        imp = np.mean([(b-t)/b*100 for b,t in zip(b_rmses, t_rmses)])
        wins = sum(1 for b,t in zip(b_rmses, t_rmses) if t < b)
        print(f"  {base:20s} -> {tr:20s}: {imp:+.2f}%, wins {wins}/{len(all_res)}")
        if gated:
            g_rmses = [r["metrics"][gated]["overall_rmse"] for r in all_res]
            imp_g = np.mean([(b-g)/b*100 for b,g in zip(b_rmses, g_rmses)])
            wins_g = sum(1 for b,g in zip(b_rmses, g_rmses) if g < b)
            print(f"  {base:20s} -> {gated:20s}: {imp_g:+.2f}%, wins {wins_g}/{len(all_res)}")


if __name__ == "__main__":
    main()

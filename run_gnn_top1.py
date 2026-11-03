"""
GNN-based Activity Cliff Residual Model on MoleculeACE
======================================================
Baseline GNN: mol graph -> GIN encoder -> readout -> MLP -> activity
Residual GNN: m(B) + g(h_B, h_A, h_B - h_A, Y_A)
  - h_B, h_A = GNN embeddings of B and neighbor A
  - structural residual in learned embedding space
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GINConv, global_mean_pool, global_add_pool
from torch.utils.data import DataLoader as TorchLoader, TensorDataset
from sklearn.metrics import mean_squared_error, mean_absolute_error
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
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
TOP_K = 1
EMB_DIM = 128  # GNN embedding dimension
NUM_GNN_LAYERS = 3

print(f"Device: {DEVICE}")


# ============================================================
# 1. Mol -> Graph conversion
# ============================================================
ATOM_FEATURES = {
    'atom_type': list(range(1, 54)),  # atomic number
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


def one_hot(val, allowed_set):
    encoding = [0] * (len(allowed_set) + 1)
    if val in allowed_set:
        encoding[allowed_set.index(val)] = 1
    else:
        encoding[-1] = 1  # unknown
    return encoding


def atom_features(atom):
    feats = []
    feats += one_hot(atom.GetAtomicNum(), ATOM_FEATURES['atom_type'])
    feats += one_hot(atom.GetDegree(), ATOM_FEATURES['degree'])
    feats += one_hot(atom.GetFormalCharge(), ATOM_FEATURES['formal_charge'])
    feats += one_hot(atom.GetHybridization(), ATOM_FEATURES['hybridization'])
    feats += one_hot(atom.GetIsAromatic(), ATOM_FEATURES['aromatic'])
    feats += one_hot(atom.GetTotalNumHs(), ATOM_FEATURES['num_hs'])
    return feats


ATOM_FEAT_DIM = (len(ATOM_FEATURES['atom_type']) + 1 +
                 len(ATOM_FEATURES['degree']) + 1 +
                 len(ATOM_FEATURES['formal_charge']) + 1 +
                 len(ATOM_FEATURES['hybridization']) + 1 +
                 len(ATOM_FEATURES['aromatic']) + 1 +
                 len(ATOM_FEATURES['num_hs']) + 1)


def mol_to_graph(smiles):
    """Convert SMILES to PyG Data object."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Node features
    x = []
    for atom in mol.GetAtoms():
        x.append(atom_features(atom))
    x = torch.FloatTensor(x)

    # Edge index
    edges = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edges.append([i, j])
        edges.append([j, i])

    if len(edges) == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    else:
        edge_index = torch.LongTensor(edges).t().contiguous()

    return Data(x=x, edge_index=edge_index)


# ============================================================
# 2. Data Loading
# ============================================================
def get_data_dir():
    spec = importlib.util.find_spec("MoleculeACE")
    return os.path.join(os.path.dirname(spec.origin), "Data", "benchmark_data")


def smiles_to_fp(smi, radius=2, nbits=FP_DIM):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
    arr = np.zeros(nbits, dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def prepare_dataset(name):
    """Load dataset and create graph + fingerprint data."""
    path = os.path.join(get_data_dir(), name)
    df = pd.read_csv(path)

    graphs, fps, ys, cliffs, splits = [], [], [], [], []
    for _, row in df.iterrows():
        g = mol_to_graph(row["smiles"])
        fp = smiles_to_fp(row["smiles"])
        if g is not None and fp is not None:
            graphs.append(g)
            fps.append(fp)
            ys.append(row["y"])
            cliffs.append(int(row["cliff_mol"]))
            splits.append(row["split"])

    fps = np.array(fps)
    ys = np.array(ys, dtype=np.float32)
    cliffs = np.array(cliffs, dtype=bool)
    splits = np.array(splits)

    tr = splits == "train"
    te = splits == "test"

    return (
        [graphs[i] for i in range(len(graphs)) if tr[i]],
        fps[tr], ys[tr], cliffs[tr],
        [graphs[i] for i in range(len(graphs)) if te[i]],
        fps[te], ys[te], cliffs[te],
    )


# ============================================================
# 3. Neighbor retrieval (still Tanimoto-based for efficiency)
# ============================================================
def tanimoto_sim(a, b):
    a_ = a.astype(bool).astype(float)
    b_ = b.astype(bool).astype(float)
    inter = a_ @ b_.T
    a_bits = a_.sum(1, keepdims=True)
    b_bits = b_.sum(1, keepdims=True).T
    union = a_bits + b_bits - inter
    return inter / np.maximum(union, 1e-8)


def get_topk_neighbor_indices(fps_q, fps_db, k=TOP_K, exclude_self=False):
    """Return indices of top-k neighbors + their similarities."""
    sim = tanimoto_sim(fps_q, fps_db)
    if exclude_self:
        np.fill_diagonal(sim, -1)
    topk_indices = np.argsort(-sim, axis=1)[:, :k]
    topk_sims = np.take_along_axis(sim, topk_indices, axis=1)
    return topk_indices, topk_sims


# ============================================================
# 4. GNN Models
# ============================================================
class GINEncoder(nn.Module):
    """GIN (Graph Isomorphism Network) encoder."""
    def __init__(self, input_dim, hidden_dim=EMB_DIM, num_layers=NUM_GNN_LAYERS,
                 dropout=0.2):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            mlp = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.convs.append(GINConv(mlp))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        self.dropout = dropout

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        # Global pooling
        h = global_add_pool(x, batch)
        return h


class BaselineGNN(nn.Module):
    """GNN encoder + MLP head for activity prediction."""
    def __init__(self, input_dim, hidden_dim=EMB_DIM):
        super().__init__()
        self.encoder = GINEncoder(input_dim, hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, data):
        h = self.encoder(data)
        return self.head(h).squeeze(-1)

    def encode(self, data):
        return self.encoder(data)


class CorrectionGNN(nn.Module):
    """Correction network operating on GNN embeddings.
    Input: h_B, h_A, h_B - h_A, Y_A, m_B
    """
    def __init__(self, emb_dim=EMB_DIM):
        super().__init__()
        # h_B (emb) + h_A (emb) + h_B-h_A (emb) + Y_A (1) + m_B (1)
        self.net = nn.Sequential(
            nn.Linear(emb_dim * 3 + 2, emb_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(emb_dim, emb_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(emb_dim // 2, 1),
        )

    def forward(self, h_b, h_a, y_a, m_b):
        residual = h_b - h_a
        x = torch.cat([h_b, h_a, residual,
                        y_a.unsqueeze(-1), m_b.unsqueeze(-1)], dim=-1)
        return self.net(x).squeeze(-1)


# ============================================================
# 5. Training Utilities
# ============================================================
def collate_graphs(graph_list):
    """Batch a list of PyG Data objects."""
    return Batch.from_data_list(graph_list)


class GraphDataset:
    def __init__(self, graphs, targets, neighbor_graphs=None, neighbor_acts=None):
        self.graphs = graphs
        self.targets = targets
        self.neighbor_graphs = neighbor_graphs
        self.neighbor_acts = neighbor_acts

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        if self.neighbor_graphs is not None:
            return (self.graphs[idx], self.targets[idx],
                    self.neighbor_graphs[idx], self.neighbor_acts[idx])
        return self.graphs[idx], self.targets[idx]


def graph_loader(dataset, batch_size=64, shuffle=True):
    """Custom dataloader for graph data."""
    n = len(dataset)
    indices = np.arange(n)
    if shuffle:
        np.random.shuffle(indices)

    for start in range(0, n, batch_size):
        batch_idx = indices[start:start + batch_size]
        batch_graphs = [dataset.graphs[i] for i in batch_idx]
        batch_y = torch.FloatTensor([dataset.targets[i] for i in batch_idx])

        batch_data = Batch.from_data_list(batch_graphs).to(DEVICE)
        batch_y = batch_y.to(DEVICE)

        if dataset.neighbor_graphs is not None:
            nb_graphs = [dataset.neighbor_graphs[i] for i in batch_idx]
            nb_data = Batch.from_data_list(nb_graphs).to(DEVICE)
            nb_acts = torch.FloatTensor(
                [dataset.neighbor_acts[i] for i in batch_idx]).to(DEVICE)
            yield batch_data, batch_y, nb_data, nb_acts
        else:
            yield batch_data, batch_y


# ============================================================
# 6. Training
# ============================================================
def train_baseline_gnn(train_graphs, y_train, val_graphs, y_val,
                       epochs=150, lr=1e-3, patience=25, batch_size=64):
    model = BaselineGNN(ATOM_FEAT_DIM).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=10, factor=0.5, min_lr=1e-5)
    criterion = nn.MSELoss()

    train_ds = GraphDataset(train_graphs, y_train)
    val_batch = Batch.from_data_list(val_graphs).to(DEVICE)
    val_y = torch.FloatTensor(y_val).to(DEVICE)

    best_val = float("inf")
    best_state = None
    wait = 0

    for epoch in range(epochs):
        model.train()
        for batch_data, batch_y in graph_loader(train_ds, batch_size):
            pred = model(batch_data)
            loss = criterion(pred, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(val_batch)
            vl = criterion(val_pred, val_y).item()
        scheduler.step(vl)

        if vl < best_val:
            best_val = vl
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    model.load_state_dict(best_state)
    return model


def get_embeddings(model, graphs, batch_size=256):
    """Extract GNN embeddings for a list of graphs."""
    model.eval()
    all_embs = []
    for start in range(0, len(graphs), batch_size):
        batch = Batch.from_data_list(graphs[start:start+batch_size]).to(DEVICE)
        with torch.no_grad():
            emb = model.encode(batch)
        all_embs.append(emb.cpu())
    return torch.cat(all_embs, dim=0)


def train_correction_gnn(correction_net, h_train, y_train, h_nb_train,
                          nb_act_train, m_train, h_val, y_val, h_nb_val,
                          nb_act_val, m_val,
                          epochs=200, lr=5e-4, patience=25, batch_size=64):
    optimizer = torch.optim.Adam(correction_net.parameters(), lr=lr,
                                  weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=10, factor=0.5, min_lr=1e-5)

    dataset = TensorDataset(h_train, torch.FloatTensor(y_train),
                             h_nb_train, torch.FloatTensor(nb_act_train),
                             torch.FloatTensor(m_train))
    loader = TorchLoader(dataset, batch_size=batch_size, shuffle=True)

    H_v = h_val.to(DEVICE)
    Y_v = torch.FloatTensor(y_val).to(DEVICE)
    HN_v = h_nb_val.to(DEVICE)
    NA_v = torch.FloatTensor(nb_act_val).to(DEVICE)
    M_v = torch.FloatTensor(m_val).to(DEVICE)

    best_val = float("inf")
    best_state = None
    wait = 0

    for epoch in range(epochs):
        correction_net.train()
        for hb, yb, hnb, nab, mb in loader:
            hb, yb, hnb, nab, mb = [t.to(DEVICE) for t in [hb, yb, hnb, nab, mb]]
            pred = mb + correction_net(hb, hnb, nab, mb)
            loss = nn.MSELoss()(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        correction_net.eval()
        with torch.no_grad():
            val_pred = M_v + correction_net(H_v, HN_v, NA_v, M_v)
            vl = nn.MSELoss()(val_pred, Y_v).item()
        scheduler.step(vl)

        if vl < best_val:
            best_val = vl
            best_state = {k: v.clone() for k, v in correction_net.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    correction_net.load_state_dict(best_state)
    return correction_net


# ============================================================
# 7. Run one dataset
# ============================================================
def run_dataset(name):
    (graphs_tr, fps_tr, y_tr, cliff_tr,
     graphs_te, fps_te, y_te, cliff_te) = prepare_dataset(name)

    if len(graphs_tr) < 20 or len(graphs_te) < 5:
        return None

    # Val split
    n = len(graphs_tr)
    perm = np.random.permutation(n)
    n_va = max(int(n * 0.15), 10)
    va_idx, tr_idx = perm[:n_va], perm[n_va:]

    graphs_va = [graphs_tr[i] for i in va_idx]
    fps_va, y_va = fps_tr[va_idx], y_tr[va_idx]

    graphs_tr2 = [graphs_tr[i] for i in tr_idx]
    fps_tr2, y_tr2 = fps_tr[tr_idx], y_tr[tr_idx]

    # Find neighbors (Tanimoto-based, indices into tr2 for train, into full tr for test)
    nb_idx_tr, _ = get_topk_neighbor_indices(fps_tr2, fps_tr2, exclude_self=True)
    nb_idx_va, _ = get_topk_neighbor_indices(fps_va, fps_tr2)
    nb_idx_te, nb_sim_te = get_topk_neighbor_indices(fps_te, fps_tr)

    # Weighted average neighbor activity (for neighbor copy baseline)
    def weighted_nb_activity(nb_idx, nb_sims, y_db):
        n = len(nb_idx)
        acts = np.zeros(n, dtype=np.float32)
        for i in range(n):
            w = nb_sims[i] / (nb_sims[i].sum() + 1e-8)
            acts[i] = (y_db[nb_idx[i]] * w).sum()
        return acts

    nb_act_te_copy = weighted_nb_activity(nb_idx_te, nb_sim_te, y_tr)

    # For neighbor graphs: use the single best neighbor
    def get_nb_graphs(nb_idx, graph_list):
        return [graph_list[nb_idx[i, 0]] for i in range(len(nb_idx))]

    def get_nb_acts(nb_idx, nb_sims, y_db):
        n = len(nb_idx)
        acts = np.zeros(n, dtype=np.float32)
        for i in range(n):
            w = nb_sims[i] / (nb_sims[i].sum() + 1e-8)
            acts[i] = (y_db[nb_idx[i]] * w).sum()
        return acts

    # ---- Model 1: Baseline GNN ----
    baseline = train_baseline_gnn(graphs_tr2, y_tr2, graphs_va, y_va)

    baseline.eval()
    with torch.no_grad():
        te_batch = Batch.from_data_list(graphs_te).to(DEVICE)
        pred_base = baseline(te_batch).cpu().numpy()

    # ---- Model 2: Two-Stage Residual GNN ----
    # Stage 1: train another baseline for m(B)
    base_m = train_baseline_gnn(graphs_tr2, y_tr2, graphs_va, y_va)

    # Get embeddings
    h_tr = get_embeddings(base_m, graphs_tr2)
    h_va = get_embeddings(base_m, graphs_va)
    h_te = get_embeddings(base_m, graphs_te)
    h_all_tr = get_embeddings(base_m, graphs_tr)  # full training set for test neighbors

    # Neighbor embeddings (weighted average of top-k)
    def weighted_nb_embeddings(nb_idx, nb_sims, h_db):
        n = len(nb_idx)
        h_nb = torch.zeros(n, h_db.shape[1])
        for i in range(n):
            w = torch.FloatTensor(nb_sims[i] / (nb_sims[i].sum() + 1e-8))
            h_nb[i] = (h_db[nb_idx[i]] * w.unsqueeze(-1)).sum(0)
        return h_nb

    _, nb_sims_tr = get_topk_neighbor_indices(fps_tr2, fps_tr2, exclude_self=True)
    _, nb_sims_va = get_topk_neighbor_indices(fps_va, fps_tr2)

    h_nb_tr = weighted_nb_embeddings(nb_idx_tr, nb_sims_tr, h_tr)
    h_nb_va = weighted_nb_embeddings(nb_idx_va, nb_sims_va, h_tr)
    h_nb_te = weighted_nb_embeddings(nb_idx_te, nb_sim_te, h_all_tr)

    nb_act_tr = get_nb_acts(nb_idx_tr, nb_sims_tr, y_tr2)
    nb_act_va = get_nb_acts(nb_idx_va, nb_sims_va, y_tr2)
    nb_act_te = get_nb_acts(nb_idx_te, nb_sim_te, y_tr)

    # m(B) predictions
    base_m.eval()
    with torch.no_grad():
        m_tr = base_m(Batch.from_data_list(graphs_tr2).to(DEVICE)).cpu().numpy()
        m_va = base_m(Batch.from_data_list(graphs_va).to(DEVICE)).cpu().numpy()
        m_te = base_m(te_batch).cpu().numpy()

    # Stage 2: train correction
    correction = CorrectionGNN(EMB_DIM).to(DEVICE)
    correction = train_correction_gnn(
        correction, h_tr, y_tr2, h_nb_tr, nb_act_tr, m_tr,
        h_va, y_va, h_nb_va, nb_act_va, m_va,
    )

    # Predict
    correction.eval()
    with torch.no_grad():
        h_te_d = h_te.to(DEVICE)
        h_nb_te_d = h_nb_te.to(DEVICE)
        na_te_d = torch.FloatTensor(nb_act_te).to(DEVICE)
        m_te_d = torch.FloatTensor(m_te).to(DEVICE)
        pred_res = (m_te_d + correction(h_te_d, h_nb_te_d, na_te_d, m_te_d)).cpu().numpy()

    pred_nb = nb_act_te_copy

    # ---- Metrics ----
    def calc(y_true, y_pred):
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r = np.corrcoef(y_true, y_pred)[0, 1] if len(y_true) > 1 else 0
        return rmse, mae, r

    results = {}
    for mname, pred in [("Baseline_GNN", pred_base),
                         ("Residual_GNN", pred_res),
                         ("NeighborCopy", pred_nb)]:
        overall = calc(y_te, pred)
        cliff_r = calc(y_te[cliff_te], pred[cliff_te]) if cliff_te.sum() > 1 else (np.nan,)*3
        noncliff_r = calc(y_te[~cliff_te], pred[~cliff_te]) if (~cliff_te).sum() > 1 else (np.nan,)*3

        results[mname] = {
            "overall_rmse": overall[0], "overall_mae": overall[1], "overall_r": overall[2],
            "cliff_rmse": cliff_r[0], "cliff_mae": cliff_r[1], "cliff_r": cliff_r[2],
            "noncliff_rmse": noncliff_r[0],
        }

    return {
        "dataset": name,
        "n_train": len(graphs_tr), "n_test": len(graphs_te),
        "n_cliff_test": int(cliff_te.sum()),
        "results": results,
    }


# ============================================================
# 8. Main
# ============================================================
def main():
    print("=" * 85)
    print("  MoleculeACE Benchmark: GNN Baseline vs GNN Residual")
    print("=" * 85)

    data_dir = get_data_dir()
    datasets = sorted([f for f in os.listdir(data_dir) if f.endswith(".csv")])

    all_results = []
    for i, dname in enumerate(datasets):
        print(f"\n[{i+1}/{len(datasets)}] {dname}...", end=" ", flush=True)
        t0 = time.time()
        try:
            res = run_dataset(dname)
            if res is None:
                print("SKIP")
                continue
            all_results.append(res)
            dt = time.time() - t0
            r = res["results"]
            print(f"({dt:.1f}s) "
                  f"GNN_Base={r['Baseline_GNN']['overall_rmse']:.3f} "
                  f"GNN_Resid={r['Residual_GNN']['overall_rmse']:.3f}")
        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()

    # ============================================================
    # Summary
    # ============================================================
    print("\n\n" + "=" * 85)
    print("  OVERALL RESULTS (RMSE)")
    print("=" * 85)
    print(f"{'Dataset':30s} | {'GNN_Base':>8s} | {'GNN_Res':>8s} | {'NB_Copy':>8s} | {'Best':>10s}")
    print("-" * 85)

    wins = {"Baseline_GNN": 0, "Residual_GNN": 0, "NeighborCopy": 0}
    cliff_wins = {"Baseline_GNN": 0, "Residual_GNN": 0}
    base_all, res_all, nb_all = [], [], []
    base_cliff, res_cliff = [], []

    for res in all_results:
        r = res["results"]
        b = r["Baseline_GNN"]["overall_rmse"]
        rv = r["Residual_GNN"]["overall_rmse"]
        n = r["NeighborCopy"]["overall_rmse"]
        base_all.append(b)
        res_all.append(rv)
        nb_all.append(n)

        best = min(r, key=lambda x: r[x]["overall_rmse"])
        wins[best] += 1
        marker = "*" if best == "Residual_GNN" else ""
        print(f"{res['dataset']:30s} | {b:8.3f} | {rv:8.3f} | {n:8.3f} | {best:>10s}{marker}")

        bc = r["Baseline_GNN"]["cliff_rmse"]
        rc = r["Residual_GNN"]["cliff_rmse"]
        if not np.isnan(bc):
            base_cliff.append(bc)
            res_cliff.append(rc)
            cliff_best = "Residual_GNN" if rc < bc else "Baseline_GNN"
            cliff_wins[cliff_best] += 1

    print("-" * 85)
    print(f"{'MEAN':30s} | {np.mean(base_all):8.3f} | {np.mean(res_all):8.3f} | "
          f"{np.mean(nb_all):8.3f} |")
    print(f"{'MEDIAN':30s} | {np.median(base_all):8.3f} | {np.median(res_all):8.3f} | "
          f"{np.median(nb_all):8.3f} |")

    print(f"\nOverall wins: {wins}")

    # Cliff table
    print("\n\n" + "=" * 85)
    print("  ACTIVITY CLIFF SUBSET (RMSE)")
    print("=" * 85)
    print(f"{'Dataset':30s} | {'n_cliff':>7s} | {'GNN_Base':>8s} | {'GNN_Res':>8s} | {'Best':>10s}")
    print("-" * 85)

    for res in all_results:
        r = res["results"]
        bc = r["Baseline_GNN"]["cliff_rmse"]
        rc = r["Residual_GNN"]["cliff_rmse"]
        if np.isnan(bc):
            continue
        best = "Residual_GNN" if rc < bc else "Baseline_GNN"
        marker = "*" if best == "Residual_GNN" else ""
        print(f"{res['dataset']:30s} | {res['n_cliff_test']:7d} | {bc:8.3f} | "
              f"{rc:8.3f} | {best:>10s}{marker}")

    if base_cliff:
        print("-" * 85)
        print(f"{'MEAN':30s} | {'':>7s} | {np.mean(base_cliff):8.3f} | "
              f"{np.mean(res_cliff):8.3f} |")
        print(f"\nCliff wins: {cliff_wins}")

    # Improvement summary
    print("\n\n" + "=" * 85)
    print("  IMPROVEMENT SUMMARY")
    print("=" * 85)

    imp = [(b - r) / b * 100 for b, r in zip(base_all, res_all)]
    print(f"\nGNN Residual vs GNN Baseline (overall RMSE % improvement):")
    print(f"  Mean: {np.mean(imp):+.2f}%  Median: {np.median(imp):+.2f}%")
    print(f"  Improved in {sum(1 for x in imp if x > 0)}/{len(imp)} datasets")

    if base_cliff:
        cimp = [(b - r) / b * 100 for b, r in zip(base_cliff, res_cliff)]
        print(f"\nGNN Residual vs GNN Baseline (CLIFF RMSE % improvement):")
        print(f"  Mean: {np.mean(cimp):+.2f}%  Median: {np.median(cimp):+.2f}%")
        print(f"  Improved in {sum(1 for x in cimp if x > 0)}/{len(cimp)} datasets")


if __name__ == "__main__":
    main()

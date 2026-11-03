"""
Advanced Residual Methods on MoleculeACE (GNN-based)
====================================================
1. RAG-style Attention: top-k neighbors with learned attention
2. Extended Residual: h_B-h_A + h_B*h_A + |h_B-h_A| + h_B*(h_B-h_A)
3. Delta Prediction: predict Y(B)-Y(A), then add Y(A)
4. Combined: RAG + Extended Residual + Delta
Baseline comparisons: GNN Baseline, Resid Only (best from ablation)
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GINConv, global_add_pool
from torch.utils.data import DataLoader, TensorDataset
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
TOP_K = 5  # for RAG attention
print(f"Device: {DEVICE}")


# ============================================================
# Mol -> Graph + FP (same as before)
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
# Neighbor retrieval: return top-k indices + sims
# ============================================================
def get_topk_indices(fps_q, fps_db, k=TOP_K, exclude_self=False):
    sim = tanimoto_sim(fps_q, fps_db)
    if exclude_self: np.fill_diagonal(sim, -1)
    indices = np.argsort(-sim, axis=1)[:, :k]
    sims = np.take_along_axis(sim, indices, axis=1)
    return indices, sims


def get_topk_embeddings(nb_idx, nb_sims, h_db, y_db):
    """Return per-neighbor embeddings (n, k, emb) and activities (n, k)."""
    n, k = nb_idx.shape
    h_nbs = torch.zeros(n, k, h_db.shape[1])
    y_nbs = np.zeros((n, k), dtype=np.float32)
    for i in range(n):
        h_nbs[i] = h_db[nb_idx[i]]
        y_nbs[i] = y_db[nb_idx[i]]
    return h_nbs, y_nbs


def get_weighted_avg(nb_idx, nb_sims, h_db, y_db):
    """Top-3 weighted average (for Resid Only baseline)."""
    n = len(nb_idx)
    h_nb = torch.zeros(n, h_db.shape[1])
    y_nb = np.zeros(n, dtype=np.float32)
    for i in range(n):
        w = nb_sims[i, :3] / (nb_sims[i, :3].sum() + 1e-8)
        wt = torch.FloatTensor(w)
        h_nb[i] = (h_db[nb_idx[i, :3]] * wt.unsqueeze(-1)).sum(0)
        y_nb[i] = (y_db[nb_idx[i, :3]] * w).sum()
    return h_nb, y_nb


# ============================================================
# GNN Encoder
# ============================================================
class GINEncoder(nn.Module):
    def __init__(self, in_dim, h=EMB_DIM, n_layers=3, drop=0.2):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for i in range(n_layers):
            d = in_dim if i==0 else h
            mlp = nn.Sequential(nn.Linear(d,h), nn.ReLU(), nn.Linear(h,h))
            self.convs.append(GINConv(mlp))
            self.bns.append(nn.BatchNorm1d(h))
        self.drop = drop

    def forward(self, data):
        x, ei, batch = data.x, data.edge_index, data.batch
        for conv, bn in zip(self.convs, self.bns):
            x = F.dropout(F.relu(bn(conv(x, ei))), self.drop, self.training)
        return global_add_pool(x, batch)


class BaselineGNN(nn.Module):
    def __init__(self, in_dim, h=EMB_DIM):
        super().__init__()
        self.encoder = GINEncoder(in_dim, h)
        self.head = nn.Sequential(nn.Linear(h, h//2), nn.ReLU(), nn.Dropout(0.2), nn.Linear(h//2, 1))

    def forward(self, data):
        return self.head(self.encoder(data)).squeeze(-1)

    def encode(self, data):
        return self.encoder(data)


# ============================================================
# Model 1: Resid Only (best from ablation) - baseline comparison
# ============================================================
class ResidOnlyNet(nn.Module):
    """h_B + (h_B - h_A) -> correction"""
    def __init__(self, emb=EMB_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb*2 + 1, emb), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(emb, emb//2), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(emb//2, 1))

    def forward(self, h_b, h_a, m_b):
        x = torch.cat([h_b, h_b - h_a, m_b.unsqueeze(-1)], -1)
        return self.net(x).squeeze(-1)


# ============================================================
# Model 2: Extended Residual
# ============================================================
class ExtendedResidNet(nn.Module):
    """h_B + (h_B-h_A) + h_B*h_A + |h_B-h_A| + h_B*(h_B-h_A) -> correction"""
    def __init__(self, emb=EMB_DIM):
        super().__init__()
        # h_B(emb) + diff(emb) + product(emb) + abs_diff(emb) + masked_diff(emb) + m_B(1)
        self.net = nn.Sequential(
            nn.Linear(emb*5 + 1, emb*2), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(emb*2, emb), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(emb, 1))

    def forward(self, h_b, h_a, m_b):
        diff = h_b - h_a
        product = h_b * h_a
        abs_diff = torch.abs(diff)
        masked_diff = h_b * diff
        x = torch.cat([h_b, diff, product, abs_diff, masked_diff,
                        m_b.unsqueeze(-1)], -1)
        return self.net(x).squeeze(-1)


# ============================================================
# Model 3: RAG Attention
# ============================================================
class RAGAttentionNet(nn.Module):
    """Attention over top-k neighbors, then correction."""
    def __init__(self, emb=EMB_DIM, k=TOP_K, n_heads=4):
        super().__init__()
        self.k = k
        # Project query (h_B) and keys (h_A_i)
        self.q_proj = nn.Linear(emb, emb)
        self.k_proj = nn.Linear(emb, emb)
        self.v_proj = nn.Linear(emb + 1, emb)  # value = h_A_i + y_A_i

        self.scale = emb ** 0.5

        # After attention: h_B + attended_context + residual -> correction
        self.net = nn.Sequential(
            nn.Linear(emb*3 + 1, emb), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(emb, emb//2), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(emb//2, 1))

    def forward(self, h_b, h_nbs, y_nbs, m_b):
        """
        h_b: (batch, emb)
        h_nbs: (batch, k, emb) - k neighbor embeddings
        y_nbs: (batch, k) - k neighbor activities
        m_b: (batch,)
        """
        Q = self.q_proj(h_b).unsqueeze(1)  # (batch, 1, emb)
        K = self.k_proj(h_nbs)              # (batch, k, emb)

        # Values: concat neighbor embedding with its activity
        v_input = torch.cat([h_nbs, y_nbs.unsqueeze(-1)], -1)  # (batch, k, emb+1)
        V = self.v_proj(v_input)            # (batch, k, emb)

        # Attention scores
        attn = (Q @ K.transpose(-2, -1)) / self.scale  # (batch, 1, k)
        attn = F.softmax(attn, dim=-1)

        # Attended context
        context = (attn @ V).squeeze(1)     # (batch, emb)

        # Weighted average neighbor for residual
        attn_w = attn.squeeze(1)            # (batch, k)
        h_a_weighted = (attn_w.unsqueeze(-1) * h_nbs).sum(1)  # (batch, emb)
        residual = h_b - h_a_weighted

        x = torch.cat([h_b, context, residual, m_b.unsqueeze(-1)], -1)
        return self.net(x).squeeze(-1)


# ============================================================
# Model 4: Delta Prediction
# ============================================================
class DeltaPredNet(nn.Module):
    """Predict Y(B) - Y(A), then add Y(A).
    Uses h_B, h_A, h_B-h_A as input."""
    def __init__(self, emb=EMB_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb*3, emb), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(emb, emb//2), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(emb//2, 1))

    def forward(self, h_b, h_a, y_a):
        """Predicts delta = Y(B) - Y(A)"""
        diff = h_b - h_a
        x = torch.cat([h_b, h_a, diff], -1)
        delta = self.net(x).squeeze(-1)
        return y_a + delta  # final prediction


# ============================================================
# Model 5: Combined (RAG + Extended Residual + Delta)
# ============================================================
class CombinedNet(nn.Module):
    """RAG attention + extended residual terms + delta prediction."""
    def __init__(self, emb=EMB_DIM, k=TOP_K):
        super().__init__()
        self.k = k
        # RAG attention
        self.q_proj = nn.Linear(emb, emb)
        self.k_proj = nn.Linear(emb, emb)
        self.v_proj = nn.Linear(emb + 1, emb)
        self.scale = emb ** 0.5

        # After attention: h_B + context + diff + product + abs_diff + y_a_weighted
        # emb + emb + emb + emb + emb + 1 = 5*emb + 1
        self.net = nn.Sequential(
            nn.Linear(emb*5 + 1, emb*2), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(emb*2, emb), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(emb, 1))

    def forward(self, h_b, h_nbs, y_nbs, m_b_unused):
        Q = self.q_proj(h_b).unsqueeze(1)
        K = self.k_proj(h_nbs)
        v_input = torch.cat([h_nbs, y_nbs.unsqueeze(-1)], -1)
        V = self.v_proj(v_input)

        attn = F.softmax((Q @ K.transpose(-2,-1)) / self.scale, dim=-1)
        context = (attn @ V).squeeze(1)

        attn_w = attn.squeeze(1)
        h_a_weighted = (attn_w.unsqueeze(-1) * h_nbs).sum(1)
        y_a_weighted = (attn_w * y_nbs).sum(1)

        diff = h_b - h_a_weighted
        product = h_b * h_a_weighted
        abs_diff = torch.abs(diff)

        x = torch.cat([h_b, context, diff, product, abs_diff,
                        y_a_weighted.unsqueeze(-1)], -1)
        delta = self.net(x).squeeze(-1)
        return y_a_weighted + delta  # delta prediction


# ============================================================
# Training
# ============================================================
def train_gnn_baseline(g_tr, y_tr, g_va, y_va, epochs=150, lr=1e-3, pat=25, bs=64):
    model = BaselineGNN(ATOM_FEAT_DIM).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=10, factor=0.5, min_lr=1e-5)
    va_b = Batch.from_data_list(g_va).to(DEVICE)
    va_y = torch.FloatTensor(y_va).to(DEVICE)
    best, bst, w = float("inf"), None, 0
    n = len(g_tr)
    for _ in range(epochs):
        model.train()
        for s in range(0, n, bs):
            idx = np.random.choice(n, min(bs, n), replace=False)
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


def train_simple_correction(net, h_tr, y_tr, h_nb_tr, m_tr,
                              h_va, y_va, h_nb_va, m_va,
                              epochs=200, lr=5e-4, pat=25, bs=64):
    """For ResidOnly and ExtendedResid (single neighbor embedding)."""
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
            pred = mb + net(hb, hnb, mb)
            loss = nn.MSELoss()(pred, yb)
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


def train_rag(net, h_tr, y_tr, h_nbs_tr, y_nbs_tr, m_tr,
              h_va, y_va, h_nbs_va, y_nbs_va, m_va,
              epochs=200, lr=5e-4, pat=25, bs=64):
    """For RAG and Combined (multi-neighbor)."""
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=10, factor=0.5, min_lr=1e-5)
    ds = TensorDataset(h_tr, torch.FloatTensor(y_tr), h_nbs_tr,
                        torch.FloatTensor(y_nbs_tr), torch.FloatTensor(m_tr))
    loader = DataLoader(ds, batch_size=bs, shuffle=True)
    H_v, Y_v = h_va.to(DEVICE), torch.FloatTensor(y_va).to(DEVICE)
    HN_v = h_nbs_va.to(DEVICE)
    YN_v = torch.FloatTensor(y_nbs_va).to(DEVICE)
    M_v = torch.FloatTensor(m_va).to(DEVICE)
    best, bst, w = float("inf"), None, 0
    for _ in range(epochs):
        net.train()
        for hb, yb, hnbs, ynbs, mb in loader:
            hb, yb, hnbs, ynbs, mb = [t.to(DEVICE) for t in [hb, yb, hnbs, ynbs, mb]]
            pred = mb + net(hb, hnbs, ynbs, mb)
            loss = nn.MSELoss()(pred, yb)
            opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            vl = nn.MSELoss()(M_v + net(H_v, HN_v, YN_v, M_v), Y_v).item()
        sched.step(vl)
        if vl < best: best, bst, w = vl, {k:v.clone() for k,v in net.state_dict().items()}, 0
        else:
            w += 1
            if w >= pat: break
    net.load_state_dict(bst); return net


def train_delta(net, h_tr, y_tr, h_nb_tr, y_nb_tr,
                h_va, y_va, h_nb_va, y_nb_va,
                epochs=200, lr=5e-4, pat=25, bs=64):
    """For Delta prediction (no m_b, predicts Y directly)."""
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
            pred = net(hb, hnb, ynb)
            loss = nn.MSELoss()(pred, yb)
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


def train_combined(net, h_tr, y_tr, h_nbs_tr, y_nbs_tr,
                    h_va, y_va, h_nbs_va, y_nbs_va,
                    epochs=200, lr=5e-4, pat=25, bs=64):
    """Combined: RAG + Extended + Delta (predicts Y directly)."""
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=10, factor=0.5, min_lr=1e-5)
    ds = TensorDataset(h_tr, torch.FloatTensor(y_tr), h_nbs_tr, torch.FloatTensor(y_nbs_tr))
    loader = DataLoader(ds, batch_size=bs, shuffle=True)
    H_v, Y_v = h_va.to(DEVICE), torch.FloatTensor(y_va).to(DEVICE)
    HN_v = h_nbs_va.to(DEVICE)
    YN_v = torch.FloatTensor(y_nbs_va).to(DEVICE)
    best, bst, w = float("inf"), None, 0
    dummy = torch.zeros(1)  # m_b unused in combined
    for _ in range(epochs):
        net.train()
        for hb, yb, hnbs, ynbs in loader:
            hb, yb, hnbs, ynbs = [t.to(DEVICE) for t in [hb, yb, hnbs, ynbs]]
            pred = net(hb, hnbs, ynbs, None)
            loss = nn.MSELoss()(pred, yb)
            opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            vl = nn.MSELoss()(net(H_v, HN_v, YN_v, None), Y_v).item()
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
    (g_tr, fps_tr, y_tr, cliff_tr,
     g_te, fps_te, y_te, cliff_te) = prepare_data(name)

    if len(g_tr) < 20 or len(g_te) < 5: return None

    n = len(g_tr)
    perm = np.random.permutation(n)
    nv = max(int(n*0.15), 10)
    vi, ti = perm[:nv], perm[nv:]

    g_va = [g_tr[i] for i in vi]; fps_va, y_va = fps_tr[vi], y_tr[vi]
    g_tr2 = [g_tr[i] for i in ti]; fps_tr2, y_tr2 = fps_tr[ti], y_tr[ti]

    # Train GNN encoder
    base = train_gnn_baseline(g_tr2, y_tr2, g_va, y_va)
    base.eval()

    # Embeddings
    h_tr = get_embeddings(base, g_tr2)
    h_va = get_embeddings(base, g_va)
    h_te = get_embeddings(base, g_te)
    h_all = get_embeddings(base, g_tr)

    with torch.no_grad():
        m_tr = base(Batch.from_data_list(g_tr2).to(DEVICE)).cpu().numpy()
        m_va = base(Batch.from_data_list(g_va).to(DEVICE)).cpu().numpy()
        m_te = base(Batch.from_data_list(g_te).to(DEVICE)).cpu().numpy()

    results = {"GNN_Baseline": m_te.copy()}

    # Top-k neighbors
    idx_tr, sim_tr = get_topk_indices(fps_tr2, fps_tr2, TOP_K, exclude_self=True)
    idx_va, sim_va = get_topk_indices(fps_va, fps_tr2, TOP_K)
    idx_te, sim_te = get_topk_indices(fps_te, fps_tr, TOP_K)

    # Single neighbor (weighted avg of top-3) for ResidOnly/ExtendedResid/Delta
    h_nb_tr, y_nb_tr = get_weighted_avg(idx_tr, sim_tr, h_tr, y_tr2)
    h_nb_va, y_nb_va = get_weighted_avg(idx_va, sim_va, h_tr, y_tr2)
    h_nb_te, y_nb_te = get_weighted_avg(idx_te, sim_te, h_all, y_tr)

    # Multi-neighbor (all k) for RAG/Combined
    h_nbs_tr, y_nbs_tr = get_topk_embeddings(idx_tr, sim_tr, h_tr, y_tr2)
    h_nbs_va, y_nbs_va = get_topk_embeddings(idx_va, sim_va, h_tr, y_tr2)
    h_nbs_te, y_nbs_te = get_topk_embeddings(idx_te, sim_te, h_all, y_tr)

    # ---- Model 1: Resid Only (previous best) ----
    net1 = ResidOnlyNet().to(DEVICE)
    net1 = train_simple_correction(net1, h_tr, y_tr2, h_nb_tr, m_tr,
                                    h_va, y_va, h_nb_va, m_va)
    net1.eval()
    with torch.no_grad():
        p = torch.FloatTensor(m_te).to(DEVICE) + net1(
            h_te.to(DEVICE), h_nb_te.to(DEVICE), torch.FloatTensor(m_te).to(DEVICE))
        results["Resid_Only"] = p.cpu().numpy()

    # ---- Model 2: Extended Residual ----
    net2 = ExtendedResidNet().to(DEVICE)
    net2 = train_simple_correction(net2, h_tr, y_tr2, h_nb_tr, m_tr,
                                    h_va, y_va, h_nb_va, m_va)
    net2.eval()
    with torch.no_grad():
        p = torch.FloatTensor(m_te).to(DEVICE) + net2(
            h_te.to(DEVICE), h_nb_te.to(DEVICE), torch.FloatTensor(m_te).to(DEVICE))
        results["Extended_Resid"] = p.cpu().numpy()

    # ---- Model 3: RAG Attention ----
    net3 = RAGAttentionNet().to(DEVICE)
    net3 = train_rag(net3, h_tr, y_tr2, h_nbs_tr, y_nbs_tr, m_tr,
                      h_va, y_va, h_nbs_va, y_nbs_va, m_va)
    net3.eval()
    with torch.no_grad():
        p = torch.FloatTensor(m_te).to(DEVICE) + net3(
            h_te.to(DEVICE), h_nbs_te.to(DEVICE),
            torch.FloatTensor(y_nbs_te).to(DEVICE),
            torch.FloatTensor(m_te).to(DEVICE))
        results["RAG_Attention"] = p.cpu().numpy()

    # ---- Model 4: Delta Prediction ----
    net4 = DeltaPredNet().to(DEVICE)
    net4 = train_delta(net4, h_tr, y_tr2, h_nb_tr, y_nb_tr,
                        h_va, y_va, h_nb_va, y_nb_va)
    net4.eval()
    with torch.no_grad():
        p = net4(h_te.to(DEVICE), h_nb_te.to(DEVICE),
                  torch.FloatTensor(y_nb_te).to(DEVICE))
        results["Delta_Pred"] = p.cpu().numpy()

    # ---- Model 5: Combined ----
    net5 = CombinedNet().to(DEVICE)
    net5 = train_combined(net5, h_tr, y_tr2, h_nbs_tr, y_nbs_tr,
                           h_va, y_va, h_nbs_va, y_nbs_va)
    net5.eval()
    with torch.no_grad():
        p = net5(h_te.to(DEVICE), h_nbs_te.to(DEVICE),
                  torch.FloatTensor(y_nbs_te).to(DEVICE), None)
        results["Combined"] = p.cpu().numpy()

    # Metrics
    def calc(yt, yp):
        return np.sqrt(mean_squared_error(yt, yp)), np.corrcoef(yt, yp)[0,1] if len(yt)>1 else 0

    out = {"dataset": name, "n_test": len(y_te), "n_cliff": int(cliff_te.sum()), "metrics": {}}
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
    print("  Advanced Methods: RAG / Extended Residual / Delta / Combined")
    print("=" * 95)

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
                  f"Base={m['GNN_Baseline']['overall_rmse']:.3f} "
                  f"ResO={m['Resid_Only']['overall_rmse']:.3f} "
                  f"Ext={m['Extended_Resid']['overall_rmse']:.3f} "
                  f"RAG={m['RAG_Attention']['overall_rmse']:.3f} "
                  f"Delta={m['Delta_Pred']['overall_rmse']:.3f} "
                  f"Comb={m['Combined']['overall_rmse']:.3f}")
        except Exception as e:
            print(f"ERROR: {e}")
            import traceback; traceback.print_exc()

    # ============================================================
    # Results
    # ============================================================
    keys = ["GNN_Baseline", "Resid_Only", "Extended_Resid",
            "RAG_Attention", "Delta_Pred", "Combined"]
    labels = {
        "GNN_Baseline": "GNN Baseline",
        "Resid_Only": "Resid Only (h_B-h_A)",
        "Extended_Resid": "Extended Resid (+product,|diff|)",
        "RAG_Attention": "RAG Attention (top-5)",
        "Delta_Pred": "Delta Pred (Y_A + delta)",
        "Combined": "Combined (RAG+Ext+Delta)",
    }

    print("\n\n" + "=" * 95)
    print("  OVERALL RESULTS")
    print("=" * 95)
    print(f"\n{'Model':42s} | {'RMSE':>8s} | {'Cliff':>8s} | {'R':>6s} | {'vs Base':>8s} | {'Win':>5s}")
    print("-" * 85)

    base_rmses = [r["metrics"]["GNN_Baseline"]["overall_rmse"] for r in all_res]
    for key in keys:
        rmses = [r["metrics"][key]["overall_rmse"] for r in all_res]
        cliffs = [r["metrics"][key]["cliff_rmse"] for r in all_res
                  if not np.isnan(r["metrics"][key]["cliff_rmse"])]
        rs = [r["metrics"][key]["overall_r"] for r in all_res]
        if key == "GNN_Baseline":
            imp, wins = "---", "---"
        else:
            diffs = [(b-v)/b*100 for b,v in zip(base_rmses, rmses)]
            imp = f"{np.mean(diffs):+.2f}%"
            wins = f"{sum(1 for d in diffs if d>0)}/{len(diffs)}"
        print(f"{labels[key]:42s} | {np.mean(rmses):8.4f} | "
              f"{np.mean(cliffs):8.4f} | {np.mean(rs):6.4f} | {imp:>8s} | {wins:>5s}")

    # Per-dataset
    short_labels = ["Base", "ResO", "ExtR", "RAG", "Delta", "Comb"]
    print(f"\n{'':28s}", end="")
    for h in short_labels:
        print(f" | {h:>6s}", end="")
    print()
    print("-" * 80)

    for r in all_res:
        m = r["metrics"]
        vals = [m[k]["overall_rmse"] for k in keys]
        bv = min(vals)
        line = f"{r['dataset']:28s}"
        for v in vals:
            mk = "*" if v == bv else " "
            line += f" | {v:5.3f}{mk}"
        print(line)

    means = [np.mean([r["metrics"][k]["overall_rmse"] for r in all_res]) for k in keys]
    bm = min(means)
    line = f"{'MEAN':28s}"
    for v in means:
        mk = "*" if v == bm else " "
        line += f" | {v:5.3f}{mk}"
    print("-" * 80)
    print(line)

    # Cliff subset
    print(f"\n[Cliff Subset RMSE]")
    print(f"{'':28s}", end="")
    for h in short_labels:
        print(f" | {h:>6s}", end="")
    print()
    print("-" * 80)

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
                mk = "*" if v == bv else " "
                line += f" | {v:5.3f}{mk}"
        print(line)

    cmeans = []
    for k in keys:
        cm = [r["metrics"][k]["cliff_rmse"] for r in all_res
              if not np.isnan(r["metrics"][k]["cliff_rmse"])]
        cmeans.append(np.mean(cm))
    bcm = min(cmeans)
    line = f"{'MEAN':28s}"
    for v in cmeans:
        mk = "*" if v == bcm else " "
        line += f" | {v:5.3f}{mk}"
    print("-" * 80)
    print(line)


if __name__ == "__main__":
    main()

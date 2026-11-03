"""
Ablation Study + Neighbor Selection Strategy Comparison
=======================================================
1) Residual feature ablation (what input components matter?)
2) Neighbor selection strategy (how to pick A?)

All on MoleculeACE benchmark, MLP-based, Top-3 neighbors.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, mean_absolute_error
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, Scaffolds
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

print(f"Device: {DEVICE}")


# ============================================================
# Data
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


def get_scaffold(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(
            mol=mol, includeChirality=False)
    except Exception:
        return None


def tanimoto_sim(a, b):
    a_ = a.astype(bool).astype(float)
    b_ = b.astype(bool).astype(float)
    inter = a_ @ b_.T
    a_bits = a_.sum(1, keepdims=True)
    b_bits = b_.sum(1, keepdims=True).T
    union = a_bits + b_bits - inter
    return inter / np.maximum(union, 1e-8)


def prepare_data(name):
    path = os.path.join(get_data_dir(), name)
    df = pd.read_csv(path)

    fps, ys, cliffs, splits, smiles_list, scaffolds = [], [], [], [], [], []
    for _, row in df.iterrows():
        fp = smiles_to_fp(row["smiles"])
        if fp is not None:
            fps.append(fp)
            ys.append(row["y"])
            cliffs.append(int(row["cliff_mol"]))
            splits.append(row["split"])
            smiles_list.append(row["smiles"])
            scaffolds.append(get_scaffold(row["smiles"]))

    fps = np.array(fps)
    ys = np.array(ys, dtype=np.float32)
    cliffs = np.array(cliffs, dtype=bool)
    splits = np.array(splits)
    scaffolds = np.array(scaffolds, dtype=object)

    tr = splits == "train"
    te = splits == "test"

    return (fps[tr], ys[tr], cliffs[tr], scaffolds[tr],
            fps[te], ys[te], cliffs[te], scaffolds[te])


# ============================================================
# Neighbor Selection Strategies
# ============================================================
def select_tanimoto_top1(fps_q, fps_db, y_db, scaff_q, scaff_db,
                          exclude_self=False):
    """Strategy 1: Most similar molecule (Tanimoto Top-1)."""
    sim = tanimoto_sim(fps_q, fps_db)
    if exclude_self:
        np.fill_diagonal(sim, -1)

    n = len(fps_q)
    nb_fps = np.zeros((n, FP_DIM), dtype=np.float32)
    nb_acts = np.zeros(n, dtype=np.float32)
    nb_sims = np.zeros(n, dtype=np.float32)

    for i in range(n):
        best = np.argmax(sim[i])
        nb_fps[i] = fps_db[best]
        nb_acts[i] = y_db[best]
        nb_sims[i] = sim[i, best]

    return nb_fps, nb_acts, nb_sims


def select_tanimoto_top3(fps_q, fps_db, y_db, scaff_q, scaff_db,
                          exclude_self=False):
    """Strategy 2: Top-3 similarity-weighted average."""
    sim = tanimoto_sim(fps_q, fps_db)
    if exclude_self:
        np.fill_diagonal(sim, -1)

    n = len(fps_q)
    nb_fps = np.zeros((n, FP_DIM), dtype=np.float32)
    nb_acts = np.zeros(n, dtype=np.float32)
    nb_sims = np.zeros(n, dtype=np.float32)

    for i in range(n):
        topk = np.argsort(-sim[i])[:3]
        w = sim[i][topk]
        w = w / (w.sum() + 1e-8)
        nb_fps[i] = (fps_db[topk] * w[:, None]).sum(0)
        nb_acts[i] = (y_db[topk] * w).sum()
        nb_sims[i] = sim[i][topk[0]]

    return nb_fps, nb_acts, nb_sims


def select_scaffold_matched(fps_q, fps_db, y_db, scaff_q, scaff_db,
                              exclude_self=False):
    """Strategy 3: Same scaffold, then most similar by Tanimoto.
    Rationale: Same scaffold = same core structure, like MMP.
    The structural residual captures R-group differences only.
    Falls back to Tanimoto Top-1 if no scaffold match exists.
    """
    sim = tanimoto_sim(fps_q, fps_db)
    if exclude_self:
        np.fill_diagonal(sim, -1)

    n = len(fps_q)
    nb_fps = np.zeros((n, FP_DIM), dtype=np.float32)
    nb_acts = np.zeros(n, dtype=np.float32)
    nb_sims = np.zeros(n, dtype=np.float32)

    for i in range(n):
        # Find molecules with same scaffold
        if scaff_q[i] is not None:
            same_scaff = np.array([j for j in range(len(fps_db))
                                    if scaff_db[j] == scaff_q[i]])
        else:
            same_scaff = np.array([], dtype=int)

        if len(same_scaff) > 0:
            # Among same scaffold, pick most similar
            scaff_sims = sim[i][same_scaff]
            best_in_scaff = same_scaff[np.argmax(scaff_sims)]
            nb_fps[i] = fps_db[best_in_scaff]
            nb_acts[i] = y_db[best_in_scaff]
            nb_sims[i] = sim[i, best_in_scaff]
        else:
            # Fallback: Tanimoto Top-1
            best = np.argmax(sim[i])
            nb_fps[i] = fps_db[best]
            nb_acts[i] = y_db[best]
            nb_sims[i] = sim[i, best]

    return nb_fps, nb_acts, nb_sims


def select_cliff_potential(fps_q, fps_db, y_db, scaff_q, scaff_db,
                            exclude_self=False):
    """Strategy 4: Among similar molecules (Tanimoto >= 0.5),
    select the one with LARGEST activity difference.
    Rationale: Directly targets activity cliff pairs.
    This neighbor is the one most likely to form a cliff with B,
    so the residual FP(B)-FP(A) encodes cliff-causing features.
    Falls back to Tanimoto Top-1 if no similar molecule exists.
    """
    sim = tanimoto_sim(fps_q, fps_db)
    if exclude_self:
        np.fill_diagonal(sim, -1)

    n = len(fps_q)
    nb_fps = np.zeros((n, FP_DIM), dtype=np.float32)
    nb_acts = np.zeros(n, dtype=np.float32)
    nb_sims = np.zeros(n, dtype=np.float32)

    for i in range(n):
        similar_mask = sim[i] >= 0.5
        similar_idx = np.where(similar_mask)[0]

        if len(similar_idx) > 0:
            # Among similar molecules, pick one with largest |Y_B - Y_A|
            # But at test time we don't know Y_B...
            # So instead: pick the one with most EXTREME activity
            # (furthest from mean), maximizing potential cliff signal
            mean_act = y_db[similar_idx].mean()
            deviations = np.abs(y_db[similar_idx] - mean_act)
            best_local = similar_idx[np.argmax(deviations)]
            nb_fps[i] = fps_db[best_local]
            nb_acts[i] = y_db[best_local]
            nb_sims[i] = sim[i, best_local]
        else:
            best = np.argmax(sim[i])
            nb_fps[i] = fps_db[best]
            nb_acts[i] = y_db[best]
            nb_sims[i] = sim[i, best]

    return nb_fps, nb_acts, nb_sims


def select_random(fps_q, fps_db, y_db, scaff_q, scaff_db,
                   exclude_self=False):
    """Strategy 5 (Control): Random neighbor from training set."""
    n = len(fps_q)
    nb_fps = np.zeros((n, FP_DIM), dtype=np.float32)
    nb_acts = np.zeros(n, dtype=np.float32)
    nb_sims = np.zeros(n, dtype=np.float32)

    sim = tanimoto_sim(fps_q, fps_db)

    for i in range(n):
        j = np.random.randint(len(fps_db))
        if exclude_self:
            while j == i:
                j = np.random.randint(len(fps_db))
        nb_fps[i] = fps_db[j]
        nb_acts[i] = y_db[j]
        nb_sims[i] = sim[i, j]

    return nb_fps, nb_acts, nb_sims


# ============================================================
# Models (Correction nets with different input configurations)
# ============================================================
class BaselineModel(nn.Module):
    def __init__(self, dim=FP_DIM, h=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, h), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(h, h // 2), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(h // 2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class CorrectionNet(nn.Module):
    """Configurable correction net for ablation."""
    def __init__(self, use_fp_a=True, use_residual=True, use_y_a=True,
                 fp_dim=FP_DIM, h=256):
        super().__init__()
        self.use_fp_a = use_fp_a
        self.use_residual = use_residual
        self.use_y_a = use_y_a

        # Calculate input dim
        in_dim = fp_dim  # FP(B) always included
        if use_fp_a:
            in_dim += fp_dim
        if use_residual:
            in_dim += fp_dim
        if use_y_a:
            in_dim += 1
        in_dim += 1  # m(B) always included

        self.net = nn.Sequential(
            nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(h, h // 2), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(h // 2, 1),
        )

    def forward(self, fp_b, fp_a, y_a, m_b):
        parts = [fp_b]
        if self.use_fp_a:
            parts.append(fp_a)
        if self.use_residual:
            parts.append(fp_b - fp_a)
        if self.use_y_a:
            parts.append(y_a.unsqueeze(-1))
        parts.append(m_b.unsqueeze(-1))
        x = torch.cat(parts, dim=-1)
        return self.net(x).squeeze(-1)


# ============================================================
# Training
# ============================================================
def train_baseline(X_tr, Y_tr, X_va, Y_va, epochs=300, lr=1e-3, patience=30):
    model = BaselineModel().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=10, factor=0.5, min_lr=1e-5)
    criterion = nn.MSELoss()

    dataset = TensorDataset(X_tr, Y_tr)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    best_val, best_state, wait = float("inf"), None, 0
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            loss = criterion(model(xb), yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            vl = criterion(model(X_va), Y_va).item()
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


def train_correction(correction, X_tr, Y_tr, NB_tr, NBA_tr, M_tr,
                      X_va, Y_va, NB_va, NBA_va, M_va,
                      epochs=300, lr=5e-4, patience=30):
    optimizer = torch.optim.Adam(correction.parameters(), lr=lr,
                                  weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=10, factor=0.5, min_lr=1e-5)

    dataset = TensorDataset(X_tr, Y_tr, NB_tr, NBA_tr, M_tr)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    best_val, best_state, wait = float("inf"), None, 0
    for _ in range(epochs):
        correction.train()
        for xb, yb, nb, nba, mb in loader:
            pred = mb + correction(xb, nb, nba, mb)
            loss = nn.MSELoss()(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        correction.eval()
        with torch.no_grad():
            val_pred = M_va + correction(X_va, NB_va, NBA_va, M_va)
            vl = nn.MSELoss()(val_pred, Y_va).item()
        scheduler.step(vl)
        if vl < best_val:
            best_val = vl
            best_state = {k: v.clone() for k, v in correction.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break
    correction.load_state_dict(best_state)
    return correction


# ============================================================
# Run one dataset with all ablations and strategies
# ============================================================
def run_dataset(name):
    (fps_tr, y_tr, cliff_tr, scaff_tr,
     fps_te, y_te, cliff_te, scaff_te) = prepare_data(name)

    if len(fps_tr) < 20 or len(fps_te) < 5:
        return None

    # Val split
    n = len(fps_tr)
    perm = np.random.permutation(n)
    n_va = max(int(n * 0.15), 10)
    va_idx, tr_idx = perm[:n_va], perm[n_va:]

    fps_va, y_va = fps_tr[va_idx], y_tr[va_idx]
    scaff_va = scaff_tr[va_idx]
    fps_tr2, y_tr2 = fps_tr[tr_idx], y_tr[tr_idx]
    scaff_tr2 = scaff_tr[tr_idx]

    t = lambda x: torch.FloatTensor(x).to(DEVICE)

    # ---- Train shared baseline m(B) ----
    X_tr, Y_tr_t = t(fps_tr2), t(y_tr2)
    X_va, Y_va_t = t(fps_va), t(y_va)
    X_te = t(fps_te)

    baseline = train_baseline(X_tr, Y_tr_t, X_va, Y_va_t)
    baseline.eval()
    with torch.no_grad():
        m_tr = baseline(X_tr).cpu().numpy()
        m_va = baseline(X_va).cpu().numpy()
        m_te = baseline(X_te).cpu().numpy()
        pred_base = m_te.copy()

    results = {}
    results["Baseline"] = pred_base

    # ---- Neighbor Selection Strategies ----
    strategies = {
        "Tanimoto_Top1": select_tanimoto_top1,
        "Tanimoto_Top3": select_tanimoto_top3,
        "Scaffold_Match": select_scaffold_matched,
        "Cliff_Potential": select_cliff_potential,
        "Random": select_random,
    }

    strategy_neighbors = {}
    for sname, sfunc in strategies.items():
        nb_tr = sfunc(fps_tr2, fps_tr2, y_tr2, scaff_tr2, scaff_tr2,
                       exclude_self=True)
        nb_va = sfunc(fps_va, fps_tr2, y_tr2, scaff_va, scaff_tr2)
        nb_te = sfunc(fps_te, fps_tr, y_tr, scaff_te, scaff_tr)
        strategy_neighbors[sname] = (nb_tr, nb_va, nb_te)

    # ---- Part 1: Ablation (using Tanimoto_Top3 as base strategy) ----
    ablation_configs = {
        "No_Neighbor":  {"use_fp_a": False, "use_residual": False, "use_y_a": False},
        "Full":         {"use_fp_a": True,  "use_residual": True,  "use_y_a": True},
        "No_Residual":  {"use_fp_a": True,  "use_residual": False, "use_y_a": True},
        "No_FP_A":      {"use_fp_a": False, "use_residual": True,  "use_y_a": True},
        "No_Y_A":       {"use_fp_a": True,  "use_residual": True,  "use_y_a": False},
        "Residual_Only":{"use_fp_a": False, "use_residual": True,  "use_y_a": False},
        "Y_A_Only":     {"use_fp_a": False, "use_residual": False, "use_y_a": True},
    }

    nb_tr, nb_va, nb_te = strategy_neighbors["Tanimoto_Top3"]
    NB_tr_t, NBA_tr_t = t(nb_tr[0]), t(nb_tr[1])
    NB_va_t, NBA_va_t = t(nb_va[0]), t(nb_va[1])
    NB_te_t, NBA_te_t = t(nb_te[0]), t(nb_te[1])
    M_tr_t, M_va_t, M_te_t = t(m_tr), t(m_va), t(m_te)

    for abl_name, cfg in ablation_configs.items():
        corr = CorrectionNet(**cfg).to(DEVICE)
        corr = train_correction(corr, X_tr, Y_tr_t, NB_tr_t, NBA_tr_t, M_tr_t,
                                 X_va, Y_va_t, NB_va_t, NBA_va_t, M_va_t)
        corr.eval()
        with torch.no_grad():
            pred = (M_te_t + corr(X_te, NB_te_t, NBA_te_t, M_te_t)).cpu().numpy()
        results[f"Abl_{abl_name}"] = pred

    # ---- Part 2: Strategy comparison (all use Full config) ----
    for sname in strategies:
        nb_tr_s, nb_va_s, nb_te_s = strategy_neighbors[sname]
        NB_s_tr, NBA_s_tr = t(nb_tr_s[0]), t(nb_tr_s[1])
        NB_s_va, NBA_s_va = t(nb_va_s[0]), t(nb_va_s[1])
        NB_s_te, NBA_s_te = t(nb_te_s[0]), t(nb_te_s[1])

        corr = CorrectionNet(use_fp_a=True, use_residual=True,
                              use_y_a=True).to(DEVICE)
        corr = train_correction(corr, X_tr, Y_tr_t, NB_s_tr, NBA_s_tr, M_tr_t,
                                 X_va, Y_va_t, NB_s_va, NBA_s_va, M_va_t)
        corr.eval()
        with torch.no_grad():
            pred = (M_te_t + corr(X_te, NB_s_te, NBA_s_te, M_te_t)).cpu().numpy()
        results[f"Strat_{sname}"] = pred

    # ---- Compute metrics ----
    def calc(y_true, y_pred):
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r = np.corrcoef(y_true, y_pred)[0, 1] if len(y_true) > 1 else 0
        return rmse, mae, r

    output = {"dataset": name, "n_test": len(fps_te),
              "n_cliff": int(cliff_te.sum()), "metrics": {}}

    for rname, pred in results.items():
        overall = calc(y_te, pred)
        cliff_m = calc(y_te[cliff_te], pred[cliff_te]) if cliff_te.sum() > 1 else (np.nan,)*3
        output["metrics"][rname] = {
            "overall_rmse": overall[0], "overall_r": overall[2],
            "cliff_rmse": cliff_m[0], "cliff_r": cliff_m[2],
        }

    return output


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 90)
    print("  Ablation Study + Neighbor Selection Strategy Comparison")
    print("=" * 90)

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
            m = res["metrics"]
            print(f"({dt:.1f}s) Base={m['Baseline']['overall_rmse']:.3f} "
                  f"NoNB={m['Abl_No_Neighbor']['overall_rmse']:.3f} "
                  f"Full={m['Abl_Full']['overall_rmse']:.3f} "
                  f"NoRes={m['Abl_No_Residual']['overall_rmse']:.3f}")
        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()

    # ============================================================
    # Part 1: Ablation Results
    # ============================================================
    abl_keys = ["Baseline", "Abl_No_Neighbor", "Abl_Full", "Abl_No_Residual",
                "Abl_No_FP_A", "Abl_No_Y_A", "Abl_Residual_Only", "Abl_Y_A_Only"]
    abl_labels = {
        "Baseline":          "Baseline m(B) only (no correction)",
        "Abl_No_Neighbor":   "m(B)+g(FP(B), m(B)) (no neighbor)",
        "Abl_Full":          "Full: FP(B)+FP(A)+Resid+Y(A)",
        "Abl_No_Residual":   "- Residual: FP(B)+FP(A)+Y(A)",
        "Abl_No_FP_A":       "- FP(A): FP(B)+Resid+Y(A)",
        "Abl_No_Y_A":        "- Y(A): FP(B)+FP(A)+Resid",
        "Abl_Residual_Only": "Resid only: FP(B)+Resid",
        "Abl_Y_A_Only":      "Y(A) only: FP(B)+Y(A)",
    }

    print("\n\n" + "=" * 90)
    print("  PART 1: FEATURE ABLATION (Tanimoto Top-3 neighbor)")
    print("=" * 90)
    print(f"\n{'Configuration':45s} | {'Overall RMSE':>12s} | {'Cliff RMSE':>10s} | {'Overall R':>9s}")
    print("-" * 90)

    for key in abl_keys:
        rmses = [r["metrics"][key]["overall_rmse"] for r in all_results]
        cliff_rmses = [r["metrics"][key]["cliff_rmse"] for r in all_results
                       if not np.isnan(r["metrics"][key]["cliff_rmse"])]
        rs = [r["metrics"][key]["overall_r"] for r in all_results]
        print(f"{abl_labels[key]:45s} | {np.mean(rmses):12.4f} | "
              f"{np.mean(cliff_rmses):10.4f} | {np.mean(rs):9.4f}")

    # Per-dataset ablation detail
    print(f"\n{'':30s} | {'Base':>6s} | {'NoNB':>6s} | {'Full':>6s} | {'NoRes':>6s} | "
          f"{'NoFPA':>6s} | {'NoYA':>6s} | {'ResOnly':>7s} | {'YAOnly':>7s}")
    print("-" * 100)

    abl_short = ["Baseline", "Abl_No_Neighbor", "Abl_Full", "Abl_No_Residual",
                 "Abl_No_FP_A", "Abl_No_Y_A", "Abl_Residual_Only", "Abl_Y_A_Only"]

    for res in all_results:
        m = res["metrics"]
        vals = [m[k]["overall_rmse"] for k in abl_short]
        best_val = min(vals)
        line = f"{res['dataset']:30s}"
        for v in vals:
            marker = "*" if v == best_val else " "
            line += f" | {v:5.3f}{marker}"
        print(line)

    means = []
    for k in abl_short:
        means.append(np.mean([r["metrics"][k]["overall_rmse"] for r in all_results]))
    line = f"{'MEAN':30s}"
    best_mean = min(means)
    for v in means:
        marker = "*" if v == best_mean else " "
        line += f" | {v:5.3f}{marker}"
    print("-" * 90)
    print(line)

    # Improvement vs baseline
    print(f"\n{'Configuration':45s} | {'vs Baseline':>11s} | {'Win/Total':>10s}")
    print("-" * 70)
    for key in abl_keys[1:]:
        diffs = []
        wins = 0
        for r in all_results:
            b = r["metrics"]["Baseline"]["overall_rmse"]
            v = r["metrics"][key]["overall_rmse"]
            diffs.append((b - v) / b * 100)
            if v < b:
                wins += 1
        print(f"{abl_labels[key]:45s} | {np.mean(diffs):+10.2f}% | {wins}/{len(all_results)}")

    # ============================================================
    # Part 2: Neighbor Selection Strategy
    # ============================================================
    strat_keys = ["Strat_Tanimoto_Top1", "Strat_Tanimoto_Top3",
                  "Strat_Scaffold_Match", "Strat_Cliff_Potential",
                  "Strat_Random"]
    strat_labels = {
        "Strat_Tanimoto_Top1": "Tanimoto Top-1",
        "Strat_Tanimoto_Top3": "Tanimoto Top-3 (weighted avg)",
        "Strat_Scaffold_Match": "Scaffold-matched + Tanimoto",
        "Strat_Cliff_Potential": "Similar + extreme activity",
        "Strat_Random": "Random (control)",
    }
    strat_rationale = {
        "Strat_Tanimoto_Top1": "Most structurally similar single molecule",
        "Strat_Tanimoto_Top3": "Smooth neighbor representation, reduce noise",
        "Strat_Scaffold_Match": "Same core -> residual = R-group difference (MMP proxy)",
        "Strat_Cliff_Potential": "Similar structure but extreme activity -> cliff-aware",
        "Strat_Random": "Control: no meaningful structural relationship",
    }

    print("\n\n" + "=" * 90)
    print("  PART 2: NEIGHBOR SELECTION STRATEGY (Full correction model)")
    print("=" * 90)

    print(f"\n{'Strategy':40s} | {'Rationale'}")
    print("-" * 90)
    for k in strat_keys:
        print(f"{strat_labels[k]:40s} | {strat_rationale[k]}")

    print(f"\n{'Strategy':40s} | {'Overall RMSE':>12s} | {'Cliff RMSE':>10s} | "
          f"{'vs Base':>8s} | {'Win':>4s}")
    print("-" * 90)

    # Include baseline for reference
    base_rmses_all = [r["metrics"]["Baseline"]["overall_rmse"] for r in all_results]
    print(f"{'Baseline (no neighbor)':40s} | {np.mean(base_rmses_all):12.4f} | "
          f"{np.mean([r['metrics']['Baseline']['cliff_rmse'] for r in all_results if not np.isnan(r['metrics']['Baseline']['cliff_rmse'])]):10.4f} | "
          f"{'---':>8s} | {'---':>4s}")

    for key in strat_keys:
        rmses = [r["metrics"][key]["overall_rmse"] for r in all_results]
        cliff_rmses = [r["metrics"][key]["cliff_rmse"] for r in all_results
                       if not np.isnan(r["metrics"][key]["cliff_rmse"])]
        diffs = []
        wins = 0
        for r in all_results:
            b = r["metrics"]["Baseline"]["overall_rmse"]
            v = r["metrics"][key]["overall_rmse"]
            diffs.append((b - v) / b * 100)
            if v < b:
                wins += 1
        print(f"{strat_labels[key]:40s} | {np.mean(rmses):12.4f} | "
              f"{np.mean(cliff_rmses):10.4f} | {np.mean(diffs):+7.2f}% | "
              f"{wins}/{len(all_results)}")

    # Per-dataset strategy comparison
    print(f"\n{'':30s} | {'Base':>6s} | {'Top1':>6s} | {'Top3':>6s} | "
          f"{'Scaff':>6s} | {'Cliff':>6s} | {'Rand':>6s}")
    print("-" * 90)

    all_strat = ["Baseline"] + strat_keys
    for res in all_results:
        m = res["metrics"]
        vals = [m[k]["overall_rmse"] for k in all_strat]
        best_val = min(vals)
        line = f"{res['dataset']:30s}"
        for v in vals:
            marker = "*" if v == best_val else " "
            line += f" | {v:5.3f}{marker}"
        print(line)

    means = [np.mean([r["metrics"][k]["overall_rmse"] for r in all_results])
             for k in all_strat]
    best_mean = min(means)
    line = f"{'MEAN':30s}"
    for v in means:
        marker = "*" if v == best_mean else " "
        line += f" | {v:5.3f}{marker}"
    print("-" * 90)
    print(line)

    # Cliff subset strategy comparison
    print(f"\n[Cliff Subset]")
    print(f"{'':30s} | {'Base':>6s} | {'Top1':>6s} | {'Top3':>6s} | "
          f"{'Scaff':>6s} | {'Cliff':>6s} | {'Rand':>6s}")
    print("-" * 90)

    for res in all_results:
        m = res["metrics"]
        if np.isnan(m["Baseline"]["cliff_rmse"]):
            continue
        vals = [m[k]["cliff_rmse"] for k in all_strat]
        best_val = min([v for v in vals if not np.isnan(v)])
        line = f"{res['dataset']:30s}"
        for v in vals:
            if np.isnan(v):
                line += f" |   nan "
            else:
                marker = "*" if v == best_val else " "
                line += f" | {v:5.3f}{marker}"
        print(line)

    cliff_means = []
    for k in all_strat:
        cm = [r["metrics"][k]["cliff_rmse"] for r in all_results
              if not np.isnan(r["metrics"][k]["cliff_rmse"])]
        cliff_means.append(np.mean(cm))
    best_cm = min(cliff_means)
    line = f"{'MEAN':30s}"
    for v in cliff_means:
        marker = "*" if v == best_cm else " "
        line += f" | {v:5.3f}{marker}"
    print("-" * 90)
    print(line)

    # ============================================================
    # Summary
    # ============================================================
    print("\n\n" + "=" * 90)
    print("  CONCLUSIONS")
    print("=" * 90)
    print("""
  ABLATION:
  - Which input components of the correction net actually matter?
  - Compare: Full model vs removing each component

  NEIGHBOR SELECTION:
  - Which strategy for choosing reference molecule A works best?
  - Scaffold-matched: same core structure (MMP proxy)
  - Cliff-potential: structurally similar but extreme activity
  - Random: control to verify neighbor quality matters
""")


if __name__ == "__main__":
    main()

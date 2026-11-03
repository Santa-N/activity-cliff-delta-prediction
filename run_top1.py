"""
Activity Cliff Residual Model on MoleculeACE Benchmark
======================================================
30 datasets, pre-defined splits and cliff annotations.
Compares: Baseline vs Two-Stage Residual vs Cliff-Aware Residual
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
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


# ============================================================
# Data
# ============================================================
def get_data_dir():
    spec = importlib.util.find_spec("MoleculeACE")
    return os.path.join(os.path.dirname(spec.origin), "Data", "benchmark_data")


def load_dataset(name):
    path = os.path.join(get_data_dir(), name)
    df = pd.read_csv(path)
    return df


def smiles_to_fp(smi, radius=2, nbits=FP_DIM):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
    arr = np.zeros(nbits, dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def prepare_data(df):
    """Featurize and split using MoleculeACE's predefined split."""
    fps, ys, cliffs, splits = [], [], [], []
    for _, row in df.iterrows():
        fp = smiles_to_fp(row["smiles"])
        if fp is not None:
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

    return (fps[tr], ys[tr], cliffs[tr],
            fps[te], ys[te], cliffs[te])


# ============================================================
# Neighbors
# ============================================================
def tanimoto_sim(a, b):
    a_ = a.astype(bool).astype(float)
    b_ = b.astype(bool).astype(float)
    inter = a_ @ b_.T
    a_bits = a_.sum(1, keepdims=True)
    b_bits = b_.sum(1, keepdims=True).T
    union = a_bits + b_bits - inter
    return inter / np.maximum(union, 1e-8)


def get_topk_neighbors(fps_q, fps_db, y_db, k=TOP_K, exclude_self=False):
    sim = tanimoto_sim(fps_q, fps_db)
    if exclude_self:
        np.fill_diagonal(sim, -1)

    n = len(fps_q)
    nb_fps = np.zeros((n, FP_DIM), dtype=np.float32)
    nb_acts = np.zeros(n, dtype=np.float32)
    nb_sims = np.zeros(n, dtype=np.float32)

    for i in range(n):
        best = np.argmax(sim[i])
        nb_fps[i] = fps_db[best]       # 실제 분자 1개 그대로
        nb_acts[i] = y_db[best]         # 그 분자의 실제 활성값
        nb_sims[i] = sim[i, best]

    return nb_fps, nb_acts, nb_sims


# ============================================================
# Models
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
    def __init__(self, dim=FP_DIM, h=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim * 3 + 2, h), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(h, h // 2), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(h // 2, 1),
        )

    def forward(self, fp_b, fp_a, y_a, m_b):
        residual = fp_b - fp_a
        x = torch.cat([fp_b, fp_a, residual,
                        y_a.unsqueeze(-1), m_b.unsqueeze(-1)], dim=-1)
        return self.net(x).squeeze(-1)


# ============================================================
# Training
# ============================================================
def train_loop(model, X_tr, Y_tr, X_va, Y_va, epochs=300, lr=1e-3,
               patience=30, batch_size=64, wd=1e-4):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=10, factor=0.5, min_lr=1e-5)
    criterion = nn.MSELoss()

    dataset = TensorDataset(X_tr, Y_tr)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    best_val = float("inf")
    best_state = None
    wait = 0

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


def train_correction_loop(model, X_tr, Y_tr, NB_tr, NBA_tr, M_tr,
                           X_va, Y_va, NB_va, NBA_va, M_va,
                           weights_tr=None, epochs=300, lr=5e-4,
                           patience=30, batch_size=64):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=10, factor=0.5, min_lr=1e-5)

    if weights_tr is None:
        weights_tr = torch.ones_like(Y_tr)

    dataset = TensorDataset(X_tr, Y_tr, NB_tr, NBA_tr, M_tr, weights_tr)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    best_val = float("inf")
    best_state = None
    wait = 0

    for _ in range(epochs):
        model.train()
        for xb, yb, nb, nba, mb, wb in loader:
            pred = mb + model(xb, nb, nba, mb)
            loss = (wb * (pred - yb) ** 2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = M_va + model(X_va, NB_va, NBA_va, M_va)
            vl = nn.MSELoss()(val_pred, Y_va).item()
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


# ============================================================
# Run one dataset
# ============================================================
def run_dataset(name):
    df = load_dataset(name)
    fps_tr, y_tr, cliff_tr, fps_te, y_te, cliff_te = prepare_data(df)

    if len(fps_tr) < 20 or len(fps_te) < 5:
        return None

    # Val split from train
    n = len(fps_tr)
    perm = np.random.permutation(n)
    n_va = max(int(n * 0.15), 10)
    va_idx, tr_idx = perm[:n_va], perm[n_va:]

    fps_va, y_va = fps_tr[va_idx], y_tr[va_idx]
    cliff_va = cliff_tr[va_idx]
    fps_tr2, y_tr2 = fps_tr[tr_idx], y_tr[tr_idx]
    cliff_tr2 = cliff_tr[tr_idx]

    # Neighbors
    nb_fp_tr, nb_act_tr, _ = get_topk_neighbors(fps_tr2, fps_tr2, y_tr2,
                                                  exclude_self=True)
    nb_fp_va, nb_act_va, _ = get_topk_neighbors(fps_va, fps_tr2, y_tr2)
    nb_fp_te, nb_act_te, nb_sim_te = get_topk_neighbors(fps_te, fps_tr,
                                                          y_tr)

    # Cliff weights for training
    sim_m = tanimoto_sim(fps_tr2, fps_tr2)
    np.fill_diagonal(sim_m, -1)
    cliff_w = np.ones(len(fps_tr2), dtype=np.float32)
    for i in range(len(fps_tr2)):
        j = np.argmax(sim_m[i])
        if sim_m[i, j] >= 0.6 and abs(y_tr2[i] - y_tr2[j]) >= 1.0:
            cliff_w[i] = 1.0 + abs(y_tr2[i] - y_tr2[j])

    # Tensors
    t = lambda x: torch.FloatTensor(x).to(DEVICE)
    X_tr, Y_tr = t(fps_tr2), t(y_tr2)
    X_va, Y_va = t(fps_va), t(y_va)
    X_te, Y_te = t(fps_te), t(y_te)
    NB_tr, NBA_tr = t(nb_fp_tr), t(nb_act_tr)
    NB_va, NBA_va = t(nb_fp_va), t(nb_act_va)
    NB_te, NBA_te = t(nb_fp_te), t(nb_act_te)
    W_tr = t(cliff_w)

    # ---- Model 1: Baseline ----
    baseline = BaselineModel().to(DEVICE)
    baseline = train_loop(baseline, X_tr, Y_tr, X_va, Y_va)

    # ---- Model 2: Two-Stage Residual ----
    base_m = BaselineModel().to(DEVICE)
    base_m = train_loop(base_m, X_tr, Y_tr, X_va, Y_va)

    base_m.eval()
    with torch.no_grad():
        M_tr = base_m(X_tr)
        M_va = base_m(X_va)

    corr_g = CorrectionNet().to(DEVICE)
    corr_g = train_correction_loop(corr_g, X_tr, Y_tr, NB_tr, NBA_tr, M_tr,
                                    X_va, Y_va, NB_va, NBA_va, M_va)

    # ---- Model 3: Cliff-Aware Residual ----
    base_c = BaselineModel().to(DEVICE)
    base_c = train_loop(base_c, X_tr, Y_tr, X_va, Y_va)

    base_c.eval()
    with torch.no_grad():
        MC_tr = base_c(X_tr)
        MC_va = base_c(X_va)

    corr_c = CorrectionNet().to(DEVICE)
    corr_c = train_correction_loop(corr_c, X_tr, Y_tr, NB_tr, NBA_tr, MC_tr,
                                    X_va, Y_va, NB_va, NBA_va, MC_va,
                                    weights_tr=W_tr)

    # ---- Evaluate ----
    baseline.eval()
    base_m.eval()
    corr_g.eval()
    base_c.eval()
    corr_c.eval()

    with torch.no_grad():
        pred_base = baseline(X_te).cpu().numpy()
        m_te = base_m(X_te)
        pred_res = (m_te + corr_g(X_te, NB_te, NBA_te, m_te)).cpu().numpy()
        mc_te = base_c(X_te)
        pred_cliff = (mc_te + corr_c(X_te, NB_te, NBA_te, mc_te)).cpu().numpy()
        pred_nb = nb_act_te.copy()

    y_np = y_te

    def calc_metrics(y_true, y_pred):
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r = np.corrcoef(y_true, y_pred)[0, 1] if len(y_true) > 1 else 0
        return rmse, mae, r

    results = {}
    for model_name, pred in [("Baseline", pred_base),
                              ("Residual", pred_res),
                              ("CliffAware", pred_cliff),
                              ("NeighborCopy", pred_nb)]:
        overall = calc_metrics(y_np, pred)

        cliff_rmse, cliff_mae, cliff_r = (np.nan, np.nan, np.nan)
        if cliff_te.sum() > 1:
            cliff_rmse, cliff_mae, cliff_r = calc_metrics(
                y_np[cliff_te], pred[cliff_te])

        noncliff_rmse, noncliff_mae, noncliff_r = (np.nan, np.nan, np.nan)
        if (~cliff_te).sum() > 1:
            noncliff_rmse, noncliff_mae, noncliff_r = calc_metrics(
                y_np[~cliff_te], pred[~cliff_te])

        results[model_name] = {
            "overall_rmse": overall[0], "overall_mae": overall[1],
            "overall_r": overall[2],
            "cliff_rmse": cliff_rmse, "cliff_mae": cliff_mae,
            "cliff_r": cliff_r,
            "noncliff_rmse": noncliff_rmse, "noncliff_mae": noncliff_mae,
            "noncliff_r": noncliff_r,
        }

    return {
        "dataset": name,
        "n_train": len(fps_tr),
        "n_test": len(fps_te),
        "n_cliff_test": int(cliff_te.sum()),
        "results": results,
    }


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 85)
    print("  MoleculeACE Benchmark: Baseline vs Residual Model")
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
                print("SKIP (too small)")
                continue
            all_results.append(res)
            dt = time.time() - t0

            r = res["results"]
            print(f"({dt:.1f}s) "
                  f"Base={r['Baseline']['overall_rmse']:.3f} "
                  f"Resid={r['Residual']['overall_rmse']:.3f} "
                  f"Cliff={r['CliffAware']['overall_rmse']:.3f}")
        except Exception as e:
            print(f"ERROR: {e}")

    # ============================================================
    # Summary Table
    # ============================================================
    print("\n\n" + "=" * 85)
    print("  OVERALL RESULTS (RMSE)")
    print("=" * 85)

    header = (f"{'Dataset':30s} | {'Base':>6s} | {'Resid':>6s} | {'Cliff':>6s} | "
              f"{'NB':>6s} | {'Best':>8s}")
    print(header)
    print("-" * 85)

    wins = {"Baseline": 0, "Residual": 0, "CliffAware": 0, "NeighborCopy": 0}
    cliff_wins = {"Baseline": 0, "Residual": 0, "CliffAware": 0, "NeighborCopy": 0}

    base_rmses, res_rmses, cliff_rmses_model, nb_rmses = [], [], [], []
    base_cliff_rmses, res_cliff_rmses, ca_cliff_rmses = [], [], []

    for res in all_results:
        r = res["results"]
        b = r["Baseline"]["overall_rmse"]
        rv = r["Residual"]["overall_rmse"]
        c = r["CliffAware"]["overall_rmse"]
        n = r["NeighborCopy"]["overall_rmse"]

        base_rmses.append(b)
        res_rmses.append(rv)
        cliff_rmses_model.append(c)
        nb_rmses.append(n)

        best_name = min(["Baseline", "Residual", "CliffAware", "NeighborCopy"],
                        key=lambda x: r[x]["overall_rmse"])
        wins[best_name] += 1

        marker = "*" if best_name in ("Residual", "CliffAware") else ""
        print(f"{res['dataset']:30s} | {b:6.3f} | {rv:6.3f} | {c:6.3f} | "
              f"{n:6.3f} | {best_name:>8s}{marker}")

        # Cliff subset
        bc = r["Baseline"]["cliff_rmse"]
        rc = r["Residual"]["cliff_rmse"]
        cc = r["CliffAware"]["cliff_rmse"]
        if not np.isnan(bc):
            base_cliff_rmses.append(bc)
            res_cliff_rmses.append(rc)
            ca_cliff_rmses.append(cc)
            best_cliff = min(["Baseline", "Residual", "CliffAware"],
                             key=lambda x: r[x]["cliff_rmse"])
            cliff_wins[best_cliff] += 1

    print("-" * 85)
    print(f"{'MEAN':30s} | {np.mean(base_rmses):6.3f} | {np.mean(res_rmses):6.3f} | "
          f"{np.mean(cliff_rmses_model):6.3f} | {np.mean(nb_rmses):6.3f} |")
    print(f"{'MEDIAN':30s} | {np.median(base_rmses):6.3f} | {np.median(res_rmses):6.3f} | "
          f"{np.median(cliff_rmses_model):6.3f} | {np.median(nb_rmses):6.3f} |")

    print(f"\nOverall wins: {wins}")
    print(f"Cliff-subset wins: {cliff_wins}")

    # Cliff-specific table
    print("\n\n" + "=" * 85)
    print("  ACTIVITY CLIFF SUBSET RESULTS (RMSE)")
    print("=" * 85)
    header2 = (f"{'Dataset':30s} | {'n_cliff':>7s} | {'Base':>6s} | "
               f"{'Resid':>6s} | {'Cliff':>6s} | {'Best':>8s}")
    print(header2)
    print("-" * 85)

    for res in all_results:
        r = res["results"]
        nc = res["n_cliff_test"]
        bc = r["Baseline"]["cliff_rmse"]
        rc = r["Residual"]["cliff_rmse"]
        cc = r["CliffAware"]["cliff_rmse"]
        if np.isnan(bc):
            continue
        best = min(["Baseline", "Residual", "CliffAware"],
                   key=lambda x: r[x]["cliff_rmse"])
        marker = "*" if best != "Baseline" else ""
        print(f"{res['dataset']:30s} | {nc:7d} | {bc:6.3f} | {rc:6.3f} | "
              f"{cc:6.3f} | {best:>8s}{marker}")

    if base_cliff_rmses:
        print("-" * 85)
        print(f"{'MEAN':30s} | {'':>7s} | {np.mean(base_cliff_rmses):6.3f} | "
              f"{np.mean(res_cliff_rmses):6.3f} | {np.mean(ca_cliff_rmses):6.3f} |")

    # Improvement summary
    print("\n\n" + "=" * 85)
    print("  IMPROVEMENT SUMMARY")
    print("=" * 85)

    res_imp = [(b - r) / b * 100 for b, r in zip(base_rmses, res_rmses)]
    ca_imp = [(b - c) / b * 100 for b, c in zip(base_rmses, cliff_rmses_model)]

    print(f"\nResidual vs Baseline (overall RMSE % improvement):")
    print(f"  Mean: {np.mean(res_imp):+.2f}%  Median: {np.median(res_imp):+.2f}%")
    print(f"  Improved in {sum(1 for x in res_imp if x > 0)}/{len(res_imp)} datasets")

    print(f"\nCliffAware vs Baseline (overall RMSE % improvement):")
    print(f"  Mean: {np.mean(ca_imp):+.2f}%  Median: {np.median(ca_imp):+.2f}%")
    print(f"  Improved in {sum(1 for x in ca_imp if x > 0)}/{len(ca_imp)} datasets")

    if base_cliff_rmses:
        res_cliff_imp = [(b - r) / b * 100
                         for b, r in zip(base_cliff_rmses, res_cliff_rmses)]
        ca_cliff_imp = [(b - c) / b * 100
                        for b, c in zip(base_cliff_rmses, ca_cliff_rmses)]
        print(f"\nResidual vs Baseline (CLIFF subset RMSE % improvement):")
        print(f"  Mean: {np.mean(res_cliff_imp):+.2f}%  Median: {np.median(res_cliff_imp):+.2f}%")
        print(f"  Improved in {sum(1 for x in res_cliff_imp if x > 0)}/{len(res_cliff_imp)} datasets")

        print(f"\nCliffAware vs Baseline (CLIFF subset RMSE % improvement):")
        print(f"  Mean: {np.mean(ca_cliff_imp):+.2f}%  Median: {np.median(ca_cliff_imp):+.2f}%")
        print(f"  Improved in {sum(1 for x in ca_cliff_imp if x > 0)}/{len(ca_cliff_imp)} datasets")


if __name__ == "__main__":
    main()

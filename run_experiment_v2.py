"""
Activity Cliff Residual Model v2 - Improved
============================================
Key improvements over v1:
1. Two-stage training: pre-train m(B), freeze, then train g
2. Top-k neighbor averaging (k=3)
3. Cliff-aware weighted loss
4. Better regularization
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit import RDLogger
import warnings
import os
import urllib.request

warnings.filterwarnings("ignore")
RDLogger.logger().setLevel(RDLogger.ERROR)

np.random.seed(42)
torch.manual_seed(42)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FP_DIM = 1024
TOP_K = 3  # Number of neighbors to average

print(f"Device: {DEVICE}")


# ============================================================
# 1. Data Loading
# ============================================================
def load_data():
    cache_path = os.path.join(os.path.dirname(__file__), "bace_data.csv")
    if not os.path.exists(cache_path):
        url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/bace.csv"
        print("Downloading BACE dataset...")
        urllib.request.urlretrieve(url, cache_path)
    df = pd.read_csv(cache_path)

    # Find columns
    smi_col = next((c for c in df.columns if "mol" in c.lower() or "smiles" in c.lower()), None)
    act_col = next((c for c in df.columns if "pic50" in c.lower() or "pIC50" in c), None)
    if smi_col is None or act_col is None:
        raise ValueError(f"Cannot find columns. Available: {list(df.columns)}")

    df = df.rename(columns={smi_col: "mol", act_col: "pIC50"})
    df = df[["mol", "pIC50"]].dropna().reset_index(drop=True)
    print(f"Loaded: {len(df)} molecules, pIC50 range: [{df.pIC50.min():.2f}, {df.pIC50.max():.2f}]")
    return df


def smiles_to_fp(smiles, radius=2, nbits=FP_DIM):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
    arr = np.zeros(nbits, dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def compute_fingerprints(df):
    fps, valid_idx = [], []
    for i, smi in enumerate(df["mol"]):
        fp = smiles_to_fp(smi)
        if fp is not None:
            fps.append(fp)
            valid_idx.append(i)
    fps = np.array(fps)
    df_valid = df.iloc[valid_idx].reset_index(drop=True)
    return df_valid, fps


# ============================================================
# 2. Neighbor Retrieval (Top-K)
# ============================================================
def tanimoto_sim(fps_a, fps_b):
    a = fps_a.astype(bool).astype(float)
    b = fps_b.astype(bool).astype(float)
    inter = a @ b.T
    a_bits = a.sum(axis=1, keepdims=True)
    b_bits = b.sum(axis=1, keepdims=True).T
    union = a_bits + b_bits - inter
    return inter / np.maximum(union, 1e-8)


def get_topk_neighbors(fps_query, fps_db, y_db, k=TOP_K, exclude_self=False):
    """Return top-k averaged neighbor FP, activity, and max similarity."""
    sim = tanimoto_sim(fps_query, fps_db)
    if exclude_self:
        np.fill_diagonal(sim, -1)

    n = len(fps_query)
    nb_fps = np.zeros((n, FP_DIM), dtype=np.float32)
    nb_acts = np.zeros(n, dtype=np.float32)
    nb_sims = np.zeros(n, dtype=np.float32)

    for i in range(n):
        topk_idx = np.argsort(-sim[i])[:k]
        topk_sims = sim[i][topk_idx]

        # Similarity-weighted average
        weights = topk_sims / (topk_sims.sum() + 1e-8)
        nb_fps[i] = (fps_db[topk_idx] * weights[:, None]).sum(axis=0)
        nb_acts[i] = (y_db[topk_idx] * weights).sum()
        nb_sims[i] = topk_sims[0]  # max similarity

    return nb_fps, nb_acts, nb_sims


# ============================================================
# 3. Activity Cliff Detection
# ============================================================
def detect_cliffs(fps_test, fps_train, y_test, y_train,
                  sim_thresh=0.6, act_thresh=1.0):
    """Identify test molecules forming activity cliffs with training set."""
    sim = tanimoto_sim(fps_test, fps_train)
    cliff_mask = np.zeros(len(fps_test), dtype=bool)

    for i in range(len(fps_test)):
        best_j = np.argmax(sim[i])
        if sim[i, best_j] >= sim_thresh and abs(y_test[i] - y_train[best_j]) >= act_thresh:
            cliff_mask[i] = True

    # If too few, relax thresholds
    if cliff_mask.sum() < 5:
        for i in range(len(fps_test)):
            best_j = np.argmax(sim[i])
            if sim[i, best_j] >= 0.4 and abs(y_test[i] - y_train[best_j]) >= 0.7:
                cliff_mask[i] = True

    return cliff_mask


# ============================================================
# 4. Models
# ============================================================
class BaselineModel(nn.Module):
    """FP -> activity"""
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
    """g(FP_B, FP_A, FP_B - FP_A, Y_A, m(B)) -> residual correction"""
    def __init__(self, dim=FP_DIM, h=256):
        super().__init__()
        # Input: FP_B (dim) + FP_A (dim) + residual (dim) + Y_A (1) + m_B (1)
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
# 5. Training
# ============================================================
def train_model(model, train_data, val_data, epochs=300, lr=1e-3, patience=40,
                batch_size=64, weight_decay=1e-4):
    """Generic training loop with early stopping."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                  weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=15, factor=0.5, min_lr=1e-5
    )
    criterion = nn.MSELoss()

    train_tensors = [torch.FloatTensor(d).to(DEVICE) for d in train_data]
    val_tensors = [torch.FloatTensor(d).to(DEVICE) for d in val_data]

    dataset = TensorDataset(*train_tensors)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    best_val = float("inf")
    best_state = None
    wait = 0

    for epoch in range(epochs):
        model.train()
        for batch in loader:
            y = batch[1]
            if len(batch) == 2:
                pred = model(batch[0])
            else:
                # Correction net: inputs are fp_b, fp_a, y_a, m_b
                pred = batch[-1] + model(batch[0], batch[2], batch[3], batch[-1])
                # Actually: target - m_b is what correction should predict
                # Let's use the full residual model formulation
                pred = batch[-1] + model(batch[0], batch[2], batch[3], batch[-1])
            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            vb = val_tensors
            if len(vb) == 2:
                val_pred = model(vb[0])
            else:
                val_pred = vb[-1] + model(vb[0], vb[2], vb[3], vb[-1])
            val_loss = criterion(val_pred, vb[1]).item()

        scheduler.step(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    model.load_state_dict(best_state)
    return model, best_val


def two_stage_train(fps_train, y_train, nb_fps_train, nb_act_train,
                    fps_val, y_val, nb_fps_val, nb_act_val):
    """
    Stage 1: Train m(B) = baseline predictor
    Stage 2: Freeze m(B), train g = correction net on residuals
    """
    # Stage 1: Baseline
    print("  Stage 1: Training baseline m(B)...")
    baseline = BaselineModel().to(DEVICE)
    baseline, _ = train_model(
        baseline,
        [fps_train, y_train],
        [fps_val, y_val],
        epochs=300, lr=1e-3
    )

    # Get m(B) predictions for train/val
    baseline.eval()
    with torch.no_grad():
        m_train = baseline(torch.FloatTensor(fps_train).to(DEVICE)).cpu().numpy()
        m_val = baseline(torch.FloatTensor(fps_val).to(DEVICE)).cpu().numpy()

    # Stage 2: Correction net
    print("  Stage 2: Training correction g(B, A, B-A)...")
    correction = CorrectionNet().to(DEVICE)

    # Train data for correction: fp_b, y, nb_fp, nb_act, m_b
    correction, _ = train_model(
        correction,
        [fps_train, y_train, nb_fps_train, nb_act_train, m_train],
        [fps_val, y_val, nb_fps_val, nb_act_val, m_val],
        epochs=300, lr=5e-4, weight_decay=1e-3
    )

    return baseline, correction


# ============================================================
# 6. Cliff-Aware Training (bonus: weighted loss)
# ============================================================
def train_cliff_aware(fps_train, y_train, nb_fps_train, nb_act_train,
                      fps_val, y_val, nb_fps_val, nb_act_val,
                      cliff_weights_train):
    """Same 2-stage but with cliff-weighted loss in stage 2."""
    # Stage 1: same baseline
    print("  Stage 1: Training baseline m(B)...")
    baseline = BaselineModel().to(DEVICE)
    baseline, _ = train_model(
        baseline, [fps_train, y_train], [fps_val, y_val],
        epochs=300, lr=1e-3
    )

    baseline.eval()
    with torch.no_grad():
        m_train = baseline(torch.FloatTensor(fps_train).to(DEVICE)).cpu().numpy()
        m_val = baseline(torch.FloatTensor(fps_val).to(DEVICE)).cpu().numpy()

    # Stage 2: Cliff-weighted correction
    print("  Stage 2: Training cliff-weighted correction...")
    correction = CorrectionNet().to(DEVICE)
    optimizer = torch.optim.Adam(correction.parameters(), lr=5e-4,
                                  weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=15, factor=0.5, min_lr=1e-5
    )

    # Tensors
    t = lambda x: torch.FloatTensor(x).to(DEVICE)
    X_tr, Y_tr = t(fps_train), t(y_train)
    NB_tr, NBA_tr, M_tr = t(nb_fps_train), t(nb_act_train), t(m_train)
    W_tr = t(cliff_weights_train)

    X_v, Y_v = t(fps_val), t(y_val)
    NB_v, NBA_v, M_v = t(nb_fps_val), t(nb_act_val), t(m_val)

    dataset = TensorDataset(X_tr, Y_tr, NB_tr, NBA_tr, M_tr, W_tr)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    best_val = float("inf")
    best_state = None
    wait = 0

    for epoch in range(300):
        correction.train()
        for xb, yb, nb, nba, mb, wb in loader:
            pred = mb + correction(xb, nb, nba, mb)
            # Weighted MSE: cliff molecules get higher weight
            loss = (wb * (pred - yb) ** 2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        correction.eval()
        with torch.no_grad():
            val_pred = M_v + correction(X_v, NB_v, NBA_v, M_v)
            val_loss = nn.MSELoss()(val_pred, Y_v).item()

        scheduler.step(val_loss)
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in correction.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= 40:
                break

    correction.load_state_dict(best_state)
    return baseline, correction


# ============================================================
# 7. Evaluation
# ============================================================
def predict_baseline(model, fps):
    model.eval()
    with torch.no_grad():
        return model(torch.FloatTensor(fps).to(DEVICE)).cpu().numpy()


def predict_residual(baseline, correction, fps, nb_fps, nb_act):
    baseline.eval()
    correction.eval()
    with torch.no_grad():
        t = lambda x: torch.FloatTensor(x).to(DEVICE)
        m = baseline(t(fps))
        corr = correction(t(fps), t(nb_fps), t(nb_act), m)
        return (m + corr).cpu().numpy(), m.cpu().numpy()


def metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r = np.corrcoef(y_true, y_pred)[0, 1] if len(y_true) > 1 else 0
    return rmse, mae, r


def print_row(label, y_true, y_pred):
    rmse, mae, r = metrics(y_true, y_pred)
    print(f"  {label:40s} | RMSE {rmse:.4f} | MAE {mae:.4f} | R {r:.4f}")
    return rmse, mae, r


# ============================================================
# 8. Main
# ============================================================
def main():
    print("=" * 75)
    print(" Activity Cliff Residual Model v2 — Improved ")
    print("=" * 75)

    # Load & featurize
    df = load_data()
    df, fps = compute_fingerprints(df)
    y = df["pIC50"].values.astype(np.float32)

    # Split
    idx = np.arange(len(fps))
    idx_tr, idx_te = train_test_split(idx, test_size=0.2, random_state=42)
    idx_tr, idx_va = train_test_split(idx_tr, test_size=0.15, random_state=42)

    fps_tr, y_tr = fps[idx_tr], y[idx_tr]
    fps_va, y_va = fps[idx_va], y[idx_va]
    fps_te, y_te = fps[idx_te], y[idx_te]
    print(f"Split: train={len(idx_tr)}, val={len(idx_va)}, test={len(idx_te)}")

    # Detect cliffs
    cliff_mask = detect_cliffs(fps_te, fps_tr, y_te, y_tr)
    n_cliff = cliff_mask.sum()
    print(f"Activity cliff test molecules: {n_cliff}/{len(fps_te)} ({100*n_cliff/len(fps_te):.1f}%)")

    # Compute cliff weights for training (based on neighbor activity difference)
    sim_tr = tanimoto_sim(fps_tr, fps_tr)
    np.fill_diagonal(sim_tr, -1)
    cliff_weights_tr = np.ones(len(fps_tr), dtype=np.float32)
    for i in range(len(fps_tr)):
        j = np.argmax(sim_tr[i])
        act_diff = abs(y_tr[i] - y_tr[j])
        if sim_tr[i, j] >= 0.6 and act_diff >= 1.0:
            cliff_weights_tr[i] = 1.0 + act_diff  # Higher weight for cliffs

    n_cliff_tr = (cliff_weights_tr > 1.0).sum()
    print(f"Cliff training molecules (weighted): {n_cliff_tr}/{len(fps_tr)}")

    # Get neighbors
    print("\nFinding top-k neighbors...")
    nb_fp_tr, nb_act_tr, _ = get_topk_neighbors(fps_tr, fps_tr, y_tr,
                                                  exclude_self=True)
    nb_fp_va, nb_act_va, _ = get_topk_neighbors(fps_va, fps_tr, y_tr)
    nb_fp_te, nb_act_te, nb_sim_te = get_topk_neighbors(fps_te, fps_tr, y_tr)
    print(f"Avg neighbor similarity (test): {nb_sim_te.mean():.3f}")

    # ---- Model 1: Baseline ----
    print("\n" + "-" * 75)
    print("Training Model 1: Baseline (FP → MLP)")
    print("-" * 75)
    baseline_only = BaselineModel().to(DEVICE)
    baseline_only, _ = train_model(
        baseline_only, [fps_tr, y_tr], [fps_va, y_va], epochs=300
    )
    pred_base = predict_baseline(baseline_only, fps_te)

    # ---- Model 2: Two-Stage Residual ----
    print("\n" + "-" * 75)
    print("Training Model 2: Two-Stage Residual")
    print("-" * 75)
    base_m, corr_g = two_stage_train(
        fps_tr, y_tr, nb_fp_tr, nb_act_tr,
        fps_va, y_va, nb_fp_va, nb_act_va,
    )
    pred_res, pred_m = predict_residual(base_m, corr_g, fps_te,
                                         nb_fp_te, nb_act_te)

    # ---- Model 3: Cliff-Aware Residual ----
    print("\n" + "-" * 75)
    print("Training Model 3: Cliff-Aware Residual")
    print("-" * 75)
    base_c, corr_c = train_cliff_aware(
        fps_tr, y_tr, nb_fp_tr, nb_act_tr,
        fps_va, y_va, nb_fp_va, nb_act_va,
        cliff_weights_tr,
    )
    pred_cliff, _ = predict_residual(base_c, corr_c, fps_te,
                                      nb_fp_te, nb_act_te)

    # Neighbor copy baseline
    pred_nb = nb_act_te.copy()

    # ============================================================
    # RESULTS
    # ============================================================
    print("\n" + "=" * 75)
    print(" RESULTS ")
    print("=" * 75)

    print(f"\n[Overall Test Set (n={len(fps_te)})]")
    r_base = print_row("Baseline (FP → MLP)", y_te, pred_base)
    r_res = print_row("Two-Stage Residual", y_te, pred_res)
    r_cliff = print_row("Cliff-Aware Residual", y_te, pred_cliff)
    r_nb = print_row("Neighbor Copy", y_te, pred_nb)

    if n_cliff > 0:
        print(f"\n[Activity Cliff Subset (n={n_cliff})]")
        rc_base = print_row("Baseline (FP → MLP)", y_te[cliff_mask], pred_base[cliff_mask])
        rc_res = print_row("Two-Stage Residual", y_te[cliff_mask], pred_res[cliff_mask])
        rc_cliff = print_row("Cliff-Aware Residual", y_te[cliff_mask], pred_cliff[cliff_mask])
        rc_nb = print_row("Neighbor Copy", y_te[cliff_mask], pred_nb[cliff_mask])

    non_cliff = ~cliff_mask
    print(f"\n[Non-Cliff Subset (n={non_cliff.sum()})]")
    print_row("Baseline (FP → MLP)", y_te[non_cliff], pred_base[non_cliff])
    print_row("Two-Stage Residual", y_te[non_cliff], pred_res[non_cliff])
    print_row("Cliff-Aware Residual", y_te[non_cliff], pred_cliff[non_cliff])
    print_row("Neighbor Copy", y_te[non_cliff], pred_nb[non_cliff])

    # Per-molecule improvement
    print("\n" + "-" * 75)
    err_base = np.abs(y_te - pred_base)
    err_res = np.abs(y_te - pred_res)
    err_cliff_model = np.abs(y_te - pred_cliff)

    imp_res = (err_res < err_base).sum()
    imp_cliff = (err_cliff_model < err_base).sum()
    print(f"Two-Stage improves:    {imp_res}/{len(y_te)} ({100*imp_res/len(y_te):.1f}%) molecules")
    print(f"Cliff-Aware improves:  {imp_cliff}/{len(y_te)} ({100*imp_cliff/len(y_te):.1f}%) molecules")

    if n_cliff > 0:
        imp_res_c = (err_res[cliff_mask] < err_base[cliff_mask]).sum()
        imp_cliff_c = (err_cliff_model[cliff_mask] < err_base[cliff_mask]).sum()
        print(f"\nOn cliff subset:")
        print(f"  Two-Stage improves:   {imp_res_c}/{n_cliff} ({100*imp_res_c/n_cliff:.1f}%)")
        print(f"  Cliff-Aware improves: {imp_cliff_c}/{n_cliff} ({100*imp_cliff_c/n_cliff:.1f}%)")

    # Example predictions
    if n_cliff > 0:
        print(f"\n{'':4s} | {'True':>7s} | {'Base':>7s} | {'Resid':>7s} | {'Cliff':>7s} | {'NB':>7s} | {'Sim':>5s}")
        print("-" * 65)
        for i in np.where(cliff_mask)[0][:12]:
            print(f"{i:4d} | {y_te[i]:7.3f} | {pred_base[i]:7.3f} | "
                  f"{pred_res[i]:7.3f} | {pred_cliff[i]:7.3f} | "
                  f"{pred_nb[i]:7.3f} | {nb_sim_te[i]:.3f}")

    # Summary
    print("\n" + "=" * 75)
    print(" SUMMARY ")
    print("=" * 75)
    print(f"""
  Model                    | Overall RMSE | Cliff RMSE | Overall R
  -------------------------+--------------+------------+----------
  Baseline (FP → MLP)      | {r_base[0]:12.4f} | {rc_base[0] if n_cliff else 'N/A':>10} | {r_base[2]:.4f}
  Two-Stage Residual       | {r_res[0]:12.4f} | {rc_res[0] if n_cliff else 'N/A':>10} | {r_res[2]:.4f}
  Cliff-Aware Residual     | {r_cliff[0]:12.4f} | {rc_cliff[0] if n_cliff else 'N/A':>10} | {r_cliff[2]:.4f}
  Neighbor Copy            | {r_nb[0]:12.4f} | {rc_nb[0] if n_cliff else 'N/A':>10} | {r_nb[2]:.4f}

  Architecture: Y_hat(B) = m(B) + g(B, FP(A), FP(B)-FP(A), Y(A))
    - m(B): pre-trained baseline (frozen in stage 2)
    - g: correction network using top-{TOP_K} neighbor info + structural residual
    - Cliff-Aware: same + weighted loss (cliff molecules get higher loss weight)
""")


if __name__ == "__main__":
    main()

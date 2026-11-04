"""
Why Delta Prediction Fails on Strong Models (SVM, RF)
=====================================================
Analysis experiments:

Part 1: Residual Pattern Analysis
  - Do strong model residuals correlate with neighbor info?
  - If residuals are already random, delta can't help.

Part 2: Use SVM prediction as reference point instead of Y(A)
  - Y_hat_SVM + delta(B, A) instead of Y(A) + delta(B, A)
  - Tests if SVM's prediction is a better anchor than neighbor activity.

Part 3: Controlled input dimension
  - Delta with same input dim as base (no feature expansion)
  - Tests if overfitting from 3x features is the cause.

Part 4: Oracle analysis
  - Give delta model the PERFECT neighbor (actual closest in activity)
  - Upper bound on how much neighbor info can help each model.

All on MoleculeACE 30 datasets.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr
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
print(f"Device: {DEVICE}")


# ============================================================
# Basics
# ============================================================
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
    fps, ys, cliffs, splits = [], [], [], []
    for _, row in df.iterrows():
        fp = smiles_to_fp(row["smiles"])
        if fp is not None:
            fps.append(fp); ys.append(row["y"])
            cliffs.append(int(row["cliff_mol"])); splits.append(row["split"])
    fps = np.array(fps); ys = np.array(ys, dtype=np.float32)
    cliffs = np.array(cliffs, dtype=bool); splits = np.array(splits)
    tr, te = splits=="train", splits=="test"
    return fps[tr], ys[tr], cliffs[tr], fps[te], ys[te], cliffs[te]

def get_neighbors(fps_q, fps_db, y_db, k=3, exclude_self=False):
    sim = tanimoto_sim(fps_q, fps_db)
    if exclude_self: np.fill_diagonal(sim, -1)
    n = len(fps_q)
    nb_fps = np.zeros((n, FP_DIM), dtype=np.float32)
    nb_ys = np.zeros(n, dtype=np.float32)
    nb_sims = np.zeros(n, dtype=np.float32)
    for i in range(n):
        topk = np.argsort(-sim[i])[:k]
        w = sim[i][topk]; w = w / (w.sum() + 1e-8)
        nb_fps[i] = (fps_db[topk] * w[:, None]).sum(0)
        nb_ys[i] = (y_db[topk] * w).sum()
        nb_sims[i] = sim[i][topk[0]]
    return nb_fps, nb_ys, nb_sims


# ============================================================
# MLP Delta (compact, for controlled experiments)
# ============================================================
class MLPDelta(nn.Module):
    def __init__(self, in_dim, h=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(h, h//2), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(h//2, 1))
    def forward(self, x):
        return self.net(x).squeeze(-1)

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
# Run one dataset
# ============================================================
def run_dataset(name):
    fps_tr, y_tr, cliff_tr, fps_te, y_te, cliff_te = prepare_data(name)
    if len(fps_tr) < 20 or len(fps_te) < 5: return None

    n = len(fps_tr); perm = np.random.permutation(n)
    nv = max(int(n*0.15), 10); vi, ti = perm[:nv], perm[nv:]
    fps_va, y_va = fps_tr[vi], y_tr[vi]
    fps_tr2, y_tr2 = fps_tr[ti], y_tr[ti]

    # Neighbors
    nb_fps_tr, nb_y_tr, _ = get_neighbors(fps_tr, fps_tr, y_tr, exclude_self=True)
    nb_fps_te, nb_y_te, nb_sims_te = get_neighbors(fps_te, fps_tr, y_tr)
    nb_fps_tr2, nb_y_tr2, _ = get_neighbors(fps_tr2, fps_tr2, y_tr2, exclude_self=True)
    nb_fps_va, nb_y_va, _ = get_neighbors(fps_va, fps_tr2, y_tr2)

    results = {}
    analysis = {}

    # ====== Base models ======
    # SVM
    svm = SVR(kernel='rbf', C=10, gamma='scale')
    svm.fit(fps_tr, y_tr)
    pred_svm = svm.predict(fps_te)
    results["SVM"] = pred_svm

    # RF
    rf = RandomForestRegressor(n_estimators=500, min_samples_leaf=2, n_jobs=-1, random_state=42)
    rf.fit(fps_tr, y_tr)
    pred_rf = rf.predict(fps_te)
    results["RF"] = pred_rf

    # MLP
    from sklearn.neural_network import MLPRegressor
    mlp_sk = MLPRegressor(hidden_layer_sizes=(256, 128), max_iter=500,
                           early_stopping=True, random_state=42)
    mlp_sk.fit(fps_tr, y_tr)
    pred_mlp = mlp_sk.predict(fps_te)
    results["MLP"] = pred_mlp

    # ================================================================
    # Part 1: Residual Pattern Analysis
    # ================================================================
    # For each model, compute residuals and check if they correlate
    # with neighbor activity difference
    for mname, pred in [("SVM", pred_svm), ("RF", pred_rf), ("MLP", pred_mlp)]:
        residuals = y_te - pred
        nb_activity_diff = nb_y_te - pred  # neighbor Y vs model prediction

        # Correlation: can neighbor info explain the residual?
        corr, pval = spearmanr(residuals, nb_activity_diff)
        # Also: correlation of residual with Y(A)
        corr_ya, _ = spearmanr(residuals, nb_y_te)
        # Residual std (how much room for improvement?)
        residual_std = np.std(residuals)

        analysis[f"{mname}_residual_corr_nb_diff"] = corr
        analysis[f"{mname}_residual_corr_ya"] = corr_ya
        analysis[f"{mname}_residual_std"] = residual_std
        analysis[f"{mname}_residual_mean_abs"] = np.mean(np.abs(residuals))

    # ================================================================
    # Part 2: SVM prediction as reference point
    # ================================================================
    # Instead of Y(A), use SVM's prediction as anchor
    # Delta learns to correct SVM's errors using neighbor info

    # SVM_pred + MLP_correction(FP_B, FP_A, FP_B-FP_A)
    svm_pred_tr = svm.predict(fps_tr2)
    svm_pred_va = svm.predict(fps_va)
    svm_pred_te = svm.predict(fps_te)

    # Target: Y - SVM_pred (correct SVM's errors)
    delta_y_tr2 = y_tr2 - svm_pred_tr
    delta_y_va = y_va - svm_pred_va

    # Features: FP_B + FP_A + diff + SVM_pred
    X_tr_svm_ref = np.concatenate([fps_tr2, nb_fps_tr2, fps_tr2 - nb_fps_tr2,
                                    svm_pred_tr.reshape(-1,1)], axis=1)
    X_va_svm_ref = np.concatenate([fps_va, nb_fps_va, fps_va - nb_fps_va,
                                    svm_pred_va.reshape(-1,1)], axis=1)
    X_te_svm_ref = np.concatenate([fps_te, nb_fps_te, fps_te - nb_fps_te,
                                    svm_pred_te.reshape(-1,1)], axis=1)

    model_svm_ref = train_mlp_delta(
        X_tr_svm_ref.shape[1], X_tr_svm_ref, delta_y_tr2,
        X_va_svm_ref, delta_y_va)
    model_svm_ref.eval()
    with torch.no_grad():
        correction = model_svm_ref(torch.FloatTensor(X_te_svm_ref).to(DEVICE)).cpu().numpy()
    results["SVM+NeuralCorrection"] = svm_pred_te + correction

    # Simpler: SVM_pred + RF_correction
    rf_corr = RandomForestRegressor(n_estimators=300, min_samples_leaf=3,
                                     n_jobs=-1, random_state=42)
    X_tr_corr = np.concatenate([fps_tr, nb_fps_tr, fps_tr - nb_fps_tr,
                                 svm.predict(fps_tr).reshape(-1,1)], axis=1)
    rf_corr.fit(X_tr_corr, y_tr - svm.predict(fps_tr))
    X_te_corr = np.concatenate([fps_te, nb_fps_te, fps_te - nb_fps_te,
                                 svm_pred_te.reshape(-1,1)], axis=1)
    results["SVM+RF_Correction"] = svm_pred_te + rf_corr.predict(X_te_corr)

    # ================================================================
    # Part 3: Controlled input dimension delta
    # ================================================================
    # Same dim as base: only use FP(B)-FP(A) as delta features (1024 dim)
    # This tests: is the failure due to 3x feature expansion?

    diff_tr2 = fps_tr2 - nb_fps_tr2
    diff_va = fps_va - nb_fps_va
    diff_te = fps_te - nb_fps_te

    # RF with just the diff features (same dim as original)
    rf_diff = RandomForestRegressor(n_estimators=500, min_samples_leaf=2,
                                     n_jobs=-1, random_state=42)
    rf_diff.fit(diff_tr2, y_tr2 - nb_y_tr2)  # train on small diff features
    results["RF+Delta_SameDim"] = nb_y_te + rf_diff.predict(diff_te)

    # SVM with just the diff features
    svm_diff = SVR(kernel='rbf', C=10, gamma='scale')
    svm_diff.fit(fps_tr - nb_fps_tr, y_tr - nb_y_tr)
    results["SVM+Delta_SameDim"] = nb_y_te + svm_diff.predict(diff_te)

    # ================================================================
    # Part 4: Oracle neighbor analysis
    # ================================================================
    # Give each test molecule the BEST possible neighbor from training set
    # (closest in actual activity, not structure)
    # This is cheating but shows upper bound

    # Oracle: for each test mol, find training mol with closest activity
    oracle_y = np.zeros(len(fps_te), dtype=np.float32)
    for i in range(len(fps_te)):
        best_j = np.argmin(np.abs(y_tr - y_te[i]))
        oracle_y[i] = y_tr[best_j]
    results["Oracle_NB_Copy"] = oracle_y

    # For SVM: how much room is there? (SVM error vs oracle)
    analysis["SVM_rmse"] = np.sqrt(mean_squared_error(y_te, pred_svm))
    analysis["RF_rmse"] = np.sqrt(mean_squared_error(y_te, pred_rf))
    analysis["MLP_rmse"] = np.sqrt(mean_squared_error(y_te, pred_mlp))
    analysis["NB_Copy_rmse"] = np.sqrt(mean_squared_error(y_te, nb_y_te))
    analysis["Oracle_NB_rmse"] = np.sqrt(mean_squared_error(y_te, oracle_y))

    # How much of SVM's error is "correctable" by neighbor info?
    svm_residuals = y_te - pred_svm
    # If we could perfectly predict the residual using only neighbor info:
    # Best case: residual = f(nb_features)
    # Fit RF on neighbor features -> residual
    rf_residual = RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=42)
    nb_feats_tr = np.concatenate([nb_fps_tr, nb_y_tr.reshape(-1,1),
                                   (fps_tr - nb_fps_tr)], axis=1)
    svm_res_tr = y_tr - svm.predict(fps_tr)
    rf_residual.fit(nb_feats_tr, svm_res_tr)
    nb_feats_te = np.concatenate([nb_fps_te, nb_y_te.reshape(-1,1),
                                   (fps_te - nb_fps_te)], axis=1)
    pred_residual = rf_residual.predict(nb_feats_te)
    results["SVM+RF_ResidualFromNB"] = pred_svm + pred_residual

    analysis["SVM_residual_predictable_rmse"] = np.sqrt(
        mean_squared_error(svm_residuals, pred_residual))

    # Metrics
    out = {"dataset": name, "n_test": len(y_te), "n_cliff": int(cliff_te.sum()),
           "metrics": {}, "analysis": analysis}
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
    print("  Why Delta Fails on Strong Models: Diagnostic Experiments")
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
            all_res.append(r); m = r["metrics"]; a = r["analysis"]
            print(f"({time.time()-t0:.1f}s) "
                  f"SVM={m['SVM']['overall_rmse']:.3f} "
                  f"SVM+NCorr={m['SVM+NeuralCorrection']['overall_rmse']:.3f} "
                  f"SVM+RFCorr={m['SVM+RF_Correction']['overall_rmse']:.3f} "
                  f"SVM_res_corr={a['SVM_residual_corr_ya']:.3f}")
        except Exception as e:
            print(f"ERROR: {e}"); import traceback; traceback.print_exc()

    # ================================================================
    # Part 1: Residual Analysis
    # ================================================================
    print("\n\n" + "=" * 110)
    print("  PART 1: RESIDUAL PATTERN ANALYSIS")
    print("  'Can neighbor info explain what each model gets wrong?'")
    print("=" * 110)

    print(f"\n{'Model':8s} | {'RMSE':>8s} | {'Resid Std':>9s} | {'|Resid|':>7s} | "
          f"{'Corr(resid,Y_A)':>15s} | {'Corr(resid,NB_diff)':>19s}")
    print("-" * 80)

    for mname in ["SVM", "RF", "MLP"]:
        rmses = [r["analysis"][f"{mname}_rmse"] for r in all_res]
        stds = [r["analysis"][f"{mname}_residual_std"] for r in all_res]
        maes = [r["analysis"][f"{mname}_residual_mean_abs"] for r in all_res]
        corr_ya = [r["analysis"][f"{mname}_residual_corr_ya"] for r in all_res]
        corr_nb = [r["analysis"][f"{mname}_residual_corr_nb_diff"] for r in all_res]
        print(f"{mname:8s} | {np.mean(rmses):8.4f} | {np.mean(stds):9.4f} | "
              f"{np.mean(maes):7.4f} | {np.mean(corr_ya):15.4f} | {np.mean(corr_nb):19.4f}")

    print("""
    Interpretation:
    - Corr(resid, Y_A): Does neighbor activity predict model errors?
      High = neighbor info is useful, Low = errors are random
    - Corr(resid, NB_diff): Does (Y_A - prediction) predict residual?
      High = delta correction can work, Low = delta adds noise
    - Resid Std: How much room for improvement?
      Small = model already good, delta has less to gain
    """)

    # ================================================================
    # Part 2: SVM as reference point
    # ================================================================
    print("\n" + "=" * 110)
    print("  PART 2: SVM PREDICTION AS REFERENCE POINT")
    print("  'Can a neural net correct SVM errors using neighbor info?'")
    print("=" * 110)

    keys2 = ["SVM", "SVM+NeuralCorrection", "SVM+RF_Correction", "SVM+RF_ResidualFromNB"]
    print(f"\n{'Model':30s} | {'RMSE':>8s} | {'Cliff':>8s} | {'vs SVM':>8s}")
    print("-" * 65)
    for k in keys2:
        rmses = [r["metrics"][k]["overall_rmse"] for r in all_res]
        cliffs = [r["metrics"][k]["cliff_rmse"] for r in all_res
                  if not np.isnan(r["metrics"][k]["cliff_rmse"])]
        svm_rmses = [r["metrics"]["SVM"]["overall_rmse"] for r in all_res]
        imp = np.mean([(s-v)/s*100 for s,v in zip(svm_rmses, rmses)])
        print(f"{k:30s} | {np.mean(rmses):8.4f} | {np.mean(cliffs):8.4f} | {imp:+7.2f}%")

    # ================================================================
    # Part 3: Same dimension delta
    # ================================================================
    print("\n" + "=" * 110)
    print("  PART 3: CONTROLLED INPUT DIMENSION")
    print("  'Is failure due to feature expansion (3x dims)?'")
    print("=" * 110)

    keys3_pairs = [("RF", "RF+Delta_SameDim"), ("SVM", "SVM+Delta_SameDim")]
    print(f"\n{'Comparison':40s} | {'Base RMSE':>10s} | {'Delta RMSE':>10s} | {'Change':>8s} | {'Win':>5s}")
    print("-" * 80)
    for base, delta in keys3_pairs:
        b_rmses = [r["metrics"][base]["overall_rmse"] for r in all_res]
        d_rmses = [r["metrics"][delta]["overall_rmse"] for r in all_res]
        imp = np.mean([(b-d)/b*100 for b,d in zip(b_rmses, d_rmses)])
        wins = sum(1 for b,d in zip(b_rmses, d_rmses) if d < b)
        print(f"{base:15s} -> {delta:20s} | {np.mean(b_rmses):10.4f} | {np.mean(d_rmses):10.4f} | "
              f"{imp:+7.2f}% | {wins}/{len(all_res)}")

    # ================================================================
    # Part 4: Oracle analysis
    # ================================================================
    print("\n" + "=" * 110)
    print("  PART 4: ORACLE ANALYSIS")
    print("  'How much room is there for neighbor-based improvement?'")
    print("=" * 110)

    keys4 = ["SVM", "RF", "MLP", "Oracle_NB_Copy"]
    print(f"\n{'Model':20s} | {'RMSE':>8s}")
    print("-" * 35)
    for k in keys4:
        rmses = [r["metrics"][k]["overall_rmse"] for r in all_res]
        print(f"{k:20s} | {np.mean(rmses):8.4f}")

    # How predictable are SVM residuals from neighbor features?
    pred_rmses = [r["analysis"]["SVM_residual_predictable_rmse"] for r in all_res]
    svm_res_stds = [r["analysis"]["SVM_residual_std"] for r in all_res]
    print(f"\nSVM residual std (average): {np.mean(svm_res_stds):.4f}")
    print(f"RF prediction of SVM residual RMSE: {np.mean(pred_rmses):.4f}")
    print(f"Ratio (pred_rmse / resid_std): {np.mean(pred_rmses)/np.mean(svm_res_stds):.4f}")
    print("""
    If ratio ~ 1.0: neighbor features CANNOT predict SVM errors
                     -> delta is useless for SVM
    If ratio < 0.5: neighbor features explain >50% of SVM variance
                     -> delta should help
    """)

    # ================================================================
    # Summary
    # ================================================================
    print("\n" + "=" * 110)
    print("  SUMMARY: WHY DELTA FAILS ON STRONG MODELS")
    print("=" * 110)

    svm_corr = np.mean([r["analysis"]["SVM_residual_corr_ya"] for r in all_res])
    rf_corr = np.mean([r["analysis"]["RF_residual_corr_ya"] for r in all_res])
    mlp_corr = np.mean([r["analysis"]["MLP_residual_corr_ya"] for r in all_res])

    print(f"""
    1. RESIDUAL CORRELATION WITH NEIGHBOR ACTIVITY:
       SVM:  {svm_corr:.4f}  (low = errors are random, delta can't help)
       RF:   {rf_corr:.4f}
       MLP:  {mlp_corr:.4f}  (higher = errors have pattern, delta can help)

    2. SVM + NEURAL CORRECTION:
       SVM base:             {np.mean([r["metrics"]["SVM"]["overall_rmse"] for r in all_res]):.4f}
       SVM + Neural Correct: {np.mean([r["metrics"]["SVM+NeuralCorrection"]["overall_rmse"] for r in all_res]):.4f}
       SVM + RF Correct:     {np.mean([r["metrics"]["SVM+RF_Correction"]["overall_rmse"] for r in all_res]):.4f}

    3. SAME-DIM DELTA (no feature expansion):
       RF base:              {np.mean([r["metrics"]["RF"]["overall_rmse"] for r in all_res]):.4f}
       RF + Delta (same dim):{np.mean([r["metrics"]["RF+Delta_SameDim"]["overall_rmse"] for r in all_res]):.4f}
       SVM base:             {np.mean([r["metrics"]["SVM"]["overall_rmse"] for r in all_res]):.4f}
       SVM+ Delta (same dim):{np.mean([r["metrics"]["SVM+Delta_SameDim"]["overall_rmse"] for r in all_res]):.4f}

    4. ORACLE (perfect neighbor):
       Oracle NB Copy:       {np.mean([r["metrics"]["Oracle_NB_Copy"]["overall_rmse"] for r in all_res]):.4f}
    """)


if __name__ == "__main__":
    main()

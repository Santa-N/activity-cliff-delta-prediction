"""
Activity Cliff Residual Model vs Baseline Comparison
=====================================================
Baseline: Fingerprint(B) → MLP → activity
Residual: m(B) + g(B, FP(A), FP(B)-FP(A)) → activity
  - A = MMP-like nearest neighbor from training set
  - m(B) = baseline prediction
  - g = residual correction using structural residual
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
import json

warnings.filterwarnings("ignore")
RDLogger.logger().setLevel(RDLogger.ERROR)

np.random.seed(42)
torch.manual_seed(42)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")


# ============================================================
# 1. Data Loading - BACE dataset from MoleculeNet
# ============================================================
def download_bace():
    """Download BACE regression dataset (pIC50 values)."""
    cache_path = os.path.join(os.path.dirname(__file__), "bace_data.csv")
    if os.path.exists(cache_path):
        return pd.read_csv(cache_path)

    # Use DeepChem's BACE dataset URL
    url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/bace.csv"
    print("Downloading BACE dataset...")
    try:
        urllib.request.urlretrieve(url, cache_path)
        df = pd.read_csv(cache_path)
        return df
    except Exception as e:
        print(f"Download failed: {e}")
        print("Generating synthetic dataset instead...")
        return None


def generate_synthetic_data(n=800):
    """Generate synthetic molecular data with activity cliffs."""
    from rdkit.Chem import MolFromSmiles, MolToSmiles

    # Use a set of real-ish SMILES with systematic modifications
    scaffolds = [
        "c1ccc2[nH]ccc2c1",      # indole
        "c1ccc2ccccc2c1",          # naphthalene
        "c1ccc(-c2ccccn2)cc1",     # phenylpyridine
        "c1ccc2c(c1)ccn2",         # indazole (approx)
        "c1ccc(-c2cccnc2)cc1",     # phenylpyridine isomer
    ]
    substituents = [
        "", "C", "CC", "CCC", "O", "N", "F", "Cl", "Br",
        "OC", "NC", "C(=O)O", "C(=O)N", "S", "C(F)(F)F",
        "OCC", "NCC", "C#N", "C(=O)OC", "[N+](=O)[O-]",
    ]

    smiles_list = []
    activities = []

    for scaffold in scaffolds:
        base_activity = np.random.uniform(4.0, 8.0)
        for sub in substituents:
            for pos_tag in ["", "c"]:
                if sub == "":
                    smi = scaffold
                else:
                    # Attach substituent
                    smi = scaffold.replace("c1ccc", f"c1c({sub})cc", 1)

                mol = MolFromSmiles(smi)
                if mol is None:
                    continue

                canonical = MolToSmiles(mol)
                if canonical in smiles_list:
                    continue

                # Activity: base + small random + occasional cliff
                noise = np.random.normal(0, 0.3)
                cliff_trigger = np.random.random()
                if cliff_trigger < 0.15:  # 15% chance of activity cliff
                    cliff_shift = np.random.choice([-1, 1]) * np.random.uniform(1.5, 3.0)
                else:
                    cliff_shift = 0

                activity = base_activity + noise + cliff_shift
                smiles_list.append(canonical)
                activities.append(activity)

    # Add more diversity
    extra_smiles = [
        "CC(=O)Oc1ccccc1C(=O)O",  # aspirin
        "CC12CCC3C(CCC4CC(O)CCC43C)C1CCC2O",  # estradiol-like
        "c1ccc(-n2ccnc2)cc1",  # phenylimidazole
        "O=C(O)c1ccccc1O",  # salicylic acid
        "Nc1ccc(O)cc1",  # aminophenol
    ]
    for smi in extra_smiles:
        mol = MolFromSmiles(smi)
        if mol:
            smiles_list.append(MolToSmiles(mol))
            activities.append(np.random.uniform(3.0, 9.0))

    df = pd.DataFrame({"mol": smiles_list, "pIC50": activities})
    df = df.drop_duplicates(subset="mol").reset_index(drop=True)
    return df


def load_data():
    """Load and prepare molecular data."""
    df = download_bace()

    if df is not None and "mol" in df.columns and "pIC50" in df.columns:
        print(f"Loaded BACE dataset: {len(df)} molecules")
        df = df[["mol", "pIC50"]].dropna().reset_index(drop=True)
    elif df is not None:
        # Try to find SMILES and activity columns
        smi_col = None
        act_col = None
        for col in df.columns:
            if "smiles" in col.lower() or col == "mol":
                smi_col = col
            if "pic50" in col.lower() or "activity" in col.lower() or col == "pIC50":
                act_col = col
        if smi_col and act_col:
            df = df.rename(columns={smi_col: "mol", act_col: "pIC50"})
            df = df[["mol", "pIC50"]].dropna().reset_index(drop=True)
            print(f"Loaded BACE dataset: {len(df)} molecules")
        else:
            print(f"Columns found: {list(df.columns)}")
            print("Falling back to synthetic data")
            df = generate_synthetic_data()
    else:
        df = generate_synthetic_data()
        print(f"Using synthetic dataset: {len(df)} molecules")

    return df


# ============================================================
# 2. Featurization
# ============================================================
def smiles_to_fp(smiles, radius=2, nbits=1024):
    """Convert SMILES to Morgan fingerprint as numpy array."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
    arr = np.zeros(nbits, dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def compute_fingerprints(df):
    """Compute fingerprints for all molecules."""
    fps = []
    valid_idx = []
    for i, smi in enumerate(df["mol"]):
        fp = smiles_to_fp(smi)
        if fp is not None:
            fps.append(fp)
            valid_idx.append(i)
    fps = np.array(fps)
    df_valid = df.iloc[valid_idx].reset_index(drop=True)
    print(f"Valid molecules with fingerprints: {len(fps)}")
    return df_valid, fps


# ============================================================
# 3. MMP-like Neighbor Retrieval
# ============================================================
def tanimoto_similarity_matrix(fps_query, fps_db):
    """Compute Tanimoto similarity between query and database fingerprints.
    Uses vectorized computation for speed.
    """
    # For binary fingerprints: Tanimoto = AB / (A + B - AB)
    # Using float arrays, interpret as binary
    q = fps_query.astype(bool).astype(float)
    d = fps_db.astype(bool).astype(float)

    intersection = q @ d.T  # (n_query, n_db)
    q_bits = q.sum(axis=1, keepdims=True)  # (n_query, 1)
    d_bits = d.sum(axis=1, keepdims=True).T  # (1, n_db)
    union = q_bits + d_bits - intersection

    sim = intersection / np.maximum(union, 1e-8)
    return sim


def find_mmp_neighbors(fps_test, fps_train, y_train, top_k=1,
                       sim_threshold=0.3):
    """For each test molecule, find the most similar training molecule
    (MMP proxy) and return its fingerprint + activity.
    """
    sim_matrix = tanimoto_similarity_matrix(fps_test, fps_train)

    neighbor_fps = []
    neighbor_activities = []
    neighbor_sims = []

    for i in range(len(fps_test)):
        sims = sim_matrix[i]
        # Get top-k neighbors above threshold
        sorted_idx = np.argsort(-sims)
        best_idx = sorted_idx[0]
        best_sim = sims[best_idx]

        if best_sim < sim_threshold:
            # If no similar neighbor, use centroid
            neighbor_fps.append(fps_train.mean(axis=0))
            neighbor_activities.append(y_train.mean())
        else:
            neighbor_fps.append(fps_train[best_idx])
            neighbor_activities.append(y_train[best_idx])
        neighbor_sims.append(best_sim)

    return (np.array(neighbor_fps), np.array(neighbor_activities),
            np.array(neighbor_sims))


def find_mmp_neighbors_train(fps_train, y_train, sim_threshold=0.3):
    """For training set, find nearest neighbor excluding self."""
    sim_matrix = tanimoto_similarity_matrix(fps_train, fps_train)
    np.fill_diagonal(sim_matrix, -1)  # exclude self

    neighbor_fps = []
    neighbor_activities = []

    for i in range(len(fps_train)):
        sims = sim_matrix[i]
        best_idx = np.argmax(sims)
        best_sim = sims[best_idx]

        if best_sim < sim_threshold:
            neighbor_fps.append(fps_train.mean(axis=0))
            neighbor_activities.append(y_train.mean())
        else:
            neighbor_fps.append(fps_train[best_idx])
            neighbor_activities.append(y_train[best_idx])

    return np.array(neighbor_fps), np.array(neighbor_activities)


# ============================================================
# 4. Models
# ============================================================
class BaselineModel(nn.Module):
    """Simple MLP: Fingerprint → activity"""
    def __init__(self, input_dim=1024, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class ResidualModel(nn.Module):
    """
    ORCA-style residual model:
    Ŷ(B) = m(B) + g(B, FP(A), FP(B) - FP(A))

    - m(B): baseline prediction from B's fingerprint
    - g: correction network using structural residual
    """
    def __init__(self, fp_dim=1024, hidden_dim=256):
        super().__init__()

        # m(B): baseline predictor
        self.baseline = nn.Sequential(
            nn.Linear(fp_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # g(B, A, B-A): residual correction
        # Input: FP(B) + FP(A) + (FP(B) - FP(A)) + activity(A)
        correction_input = fp_dim * 3 + 1
        self.correction = nn.Sequential(
            nn.Linear(correction_input, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, fp_b, fp_a, activity_a):
        """
        fp_b: target molecule fingerprint
        fp_a: MMP neighbor fingerprint
        activity_a: known activity of neighbor A
        """
        # Baseline prediction
        m_b = self.baseline(fp_b).squeeze(-1)

        # Structural residual
        residual = fp_b - fp_a

        # Correction input: B features + A features + structural residual + A's activity
        g_input = torch.cat([fp_b, fp_a, residual, activity_a.unsqueeze(-1)],
                            dim=-1)
        correction = self.correction(g_input).squeeze(-1)

        return m_b + correction

    def get_baseline(self, fp_b):
        return self.baseline(fp_b).squeeze(-1)


# ============================================================
# 5. Training
# ============================================================
def train_baseline(fps_train, y_train, fps_val, y_val,
                   epochs=200, lr=1e-3, batch_size=64):
    """Train baseline model."""
    model = BaselineModel(fps_train.shape[1]).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.MSELoss()

    X_train = torch.FloatTensor(fps_train).to(DEVICE)
    Y_train = torch.FloatTensor(y_train).to(DEVICE)
    X_val = torch.FloatTensor(fps_val).to(DEVICE)
    Y_val = torch.FloatTensor(y_val).to(DEVICE)

    dataset = TensorDataset(X_train, Y_train)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        for xb, yb in loader:
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_loss = criterion(val_pred, Y_val).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 30:
                break

    model.load_state_dict(best_state)
    return model


def train_residual(fps_train, y_train, nb_fps_train, nb_act_train,
                   fps_val, y_val, nb_fps_val, nb_act_val,
                   epochs=200, lr=1e-3, batch_size=64):
    """Train residual model."""
    model = ResidualModel(fps_train.shape[1]).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.MSELoss()

    X_train = torch.FloatTensor(fps_train).to(DEVICE)
    Y_train = torch.FloatTensor(y_train).to(DEVICE)
    NB_fp_train = torch.FloatTensor(nb_fps_train).to(DEVICE)
    NB_act_train = torch.FloatTensor(nb_act_train).to(DEVICE)

    X_val = torch.FloatTensor(fps_val).to(DEVICE)
    Y_val = torch.FloatTensor(y_val).to(DEVICE)
    NB_fp_val = torch.FloatTensor(nb_fps_val).to(DEVICE)
    NB_act_val = torch.FloatTensor(nb_act_val).to(DEVICE)

    dataset = TensorDataset(X_train, Y_train, NB_fp_train, NB_act_train)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        for xb, yb, nb_fp, nb_act in loader:
            pred = model(xb, nb_fp, nb_act)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val, NB_fp_val, NB_act_val)
            val_loss = criterion(val_pred, Y_val).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 30:
                break

    model.load_state_dict(best_state)
    return model


# ============================================================
# 6. Evaluation
# ============================================================
def identify_activity_cliffs(fps, activities, sim_threshold=0.7,
                             act_threshold=1.5):
    """Identify activity cliff molecules:
    Molecules that have a structurally similar neighbor (Tanimoto >= sim_threshold)
    but large activity difference (>= act_threshold).
    """
    sim_matrix = tanimoto_similarity_matrix(fps, fps)
    np.fill_diagonal(sim_matrix, 0)

    cliff_indices = set()
    for i in range(len(fps)):
        for j in range(i + 1, len(fps)):
            if (sim_matrix[i, j] >= sim_threshold and
                    abs(activities[i] - activities[j]) >= act_threshold):
                cliff_indices.add(i)
                cliff_indices.add(j)

    return list(cliff_indices)


def evaluate(model, fps, y, nb_fps=None, nb_act=None, is_residual=False):
    """Evaluate model and return predictions."""
    model.eval()
    with torch.no_grad():
        X = torch.FloatTensor(fps).to(DEVICE)
        if is_residual:
            NB_fp = torch.FloatTensor(nb_fps).to(DEVICE)
            NB_act = torch.FloatTensor(nb_act).to(DEVICE)
            preds = model(X, NB_fp, NB_act).cpu().numpy()
        else:
            preds = model(X).cpu().numpy()
    return preds


def print_metrics(y_true, y_pred, label=""):
    """Print evaluation metrics."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    corr = np.corrcoef(y_true, y_pred)[0, 1] if len(y_true) > 1 else 0
    print(f"  {label:30s} | RMSE: {rmse:.4f} | MAE: {mae:.4f} | R: {corr:.4f}")
    return {"rmse": rmse, "mae": mae, "r": corr}


# ============================================================
# 7. Main Experiment
# ============================================================
def main():
    print("=" * 70)
    print("Activity Cliff Residual Model Experiment")
    print("=" * 70)

    # Load data
    df = load_data()
    df, fps = compute_fingerprints(df)
    y = df["pIC50"].values.astype(np.float32)

    print(f"Activity range: {y.min():.2f} ~ {y.max():.2f}")
    print(f"Activity std: {y.std():.2f}")

    # Split data
    idx = np.arange(len(fps))
    idx_train, idx_test = train_test_split(idx, test_size=0.2, random_state=42)
    idx_train, idx_val = train_test_split(idx_train, test_size=0.15,
                                          random_state=42)

    fps_train, y_train = fps[idx_train], y[idx_train]
    fps_val, y_val = fps[idx_val], y[idx_val]
    fps_test, y_test = fps[idx_test], y[idx_test]

    print(f"\nSplit: train={len(idx_train)}, val={len(idx_val)}, "
          f"test={len(idx_test)}")

    # Identify activity cliffs in test set (relative to training set)
    print("\nIdentifying activity cliffs in test set...")
    fps_all_eval = np.vstack([fps_train, fps_test])
    y_all_eval = np.concatenate([y_train, y_test])

    # Find cliffs: test molecules that form cliffs with training molecules
    sim_test_train = tanimoto_similarity_matrix(fps_test, fps_train)
    cliff_mask = np.zeros(len(fps_test), dtype=bool)

    for i in range(len(fps_test)):
        max_sim_idx = np.argmax(sim_test_train[i])
        max_sim = sim_test_train[i, max_sim_idx]
        act_diff = abs(y_test[i] - y_train[max_sim_idx])
        if max_sim >= 0.6 and act_diff >= 1.0:
            cliff_mask[i] = True

    n_cliff = cliff_mask.sum()
    print(f"Activity cliff molecules in test set: {n_cliff}/{len(fps_test)} "
          f"({100*n_cliff/len(fps_test):.1f}%)")

    if n_cliff < 3:
        # Relax thresholds
        for i in range(len(fps_test)):
            max_sim_idx = np.argmax(sim_test_train[i])
            max_sim = sim_test_train[i, max_sim_idx]
            act_diff = abs(y_test[i] - y_train[max_sim_idx])
            if max_sim >= 0.4 and act_diff >= 0.8:
                cliff_mask[i] = True
        n_cliff = cliff_mask.sum()
        print(f"(Relaxed thresholds) Cliff molecules: {n_cliff}/{len(fps_test)}")

    # Find MMP-like neighbors
    print("\nFinding MMP-like neighbors...")
    nb_fps_train, nb_act_train = find_mmp_neighbors_train(fps_train, y_train)
    nb_fps_val, nb_act_val, _ = find_mmp_neighbors(fps_val, fps_train, y_train)
    nb_fps_test, nb_act_test, nb_sims_test = find_mmp_neighbors(
        fps_test, fps_train, y_train
    )

    print(f"Average neighbor similarity (test): {nb_sims_test.mean():.3f}")

    # Train models
    print("\n" + "-" * 70)
    print("Training Baseline Model (FP → activity)...")
    baseline = train_baseline(fps_train, y_train, fps_val, y_val)

    print("Training Residual Model (FP + MMP residual → activity)...")
    residual = train_residual(
        fps_train, y_train, nb_fps_train, nb_act_train,
        fps_val, y_val, nb_fps_val, nb_act_val,
    )

    # Evaluate
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    # Overall test performance
    pred_baseline = evaluate(baseline, fps_test, y_test)
    pred_residual = evaluate(residual, fps_test, y_test,
                             nb_fps_test, nb_act_test, is_residual=True)

    # Simple neighbor baseline: just use neighbor's activity
    pred_neighbor = nb_act_test.copy()

    print("\n[Overall Test Set Performance]")
    m_base = print_metrics(y_test, pred_baseline, "Baseline (FP → MLP)")
    m_res = print_metrics(y_test, pred_residual, "Residual (FP + MMP residual)")
    m_nb = print_metrics(y_test, pred_neighbor, "Neighbor Copy (Y_A)")

    # Activity cliff subset
    if n_cliff > 0:
        print(f"\n[Activity Cliff Subset (n={n_cliff})]")
        m_base_c = print_metrics(y_test[cliff_mask], pred_baseline[cliff_mask],
                                 "Baseline (FP → MLP)")
        m_res_c = print_metrics(y_test[cliff_mask], pred_residual[cliff_mask],
                                "Residual (FP + MMP residual)")
        m_nb_c = print_metrics(y_test[cliff_mask], pred_neighbor[cliff_mask],
                               "Neighbor Copy (Y_A)")

    # Non-cliff subset
    non_cliff_mask = ~cliff_mask
    if non_cliff_mask.sum() > 0:
        print(f"\n[Non-Cliff Subset (n={non_cliff_mask.sum()})]")
        print_metrics(y_test[non_cliff_mask], pred_baseline[non_cliff_mask],
                      "Baseline (FP → MLP)")
        print_metrics(y_test[non_cliff_mask], pred_residual[non_cliff_mask],
                      "Residual (FP + MMP residual)")
        print_metrics(y_test[non_cliff_mask], pred_neighbor[non_cliff_mask],
                      "Neighbor Copy (Y_A)")

    # Improvement analysis
    print("\n" + "=" * 70)
    print("IMPROVEMENT ANALYSIS")
    print("=" * 70)

    base_err = np.abs(y_test - pred_baseline)
    res_err = np.abs(y_test - pred_residual)
    improved = (res_err < base_err).sum()
    total = len(y_test)
    print(f"\nResidual model improves {improved}/{total} "
          f"({100*improved/total:.1f}%) test molecules")

    if n_cliff > 0:
        base_err_c = np.abs(y_test[cliff_mask] - pred_baseline[cliff_mask])
        res_err_c = np.abs(y_test[cliff_mask] - pred_residual[cliff_mask])
        improved_c = (res_err_c < base_err_c).sum()
        print(f"On cliff molecules: {improved_c}/{n_cliff} "
              f"({100*improved_c/n_cliff:.1f}%) improved")

    overall_rmse_gain = m_base["rmse"] - m_res["rmse"]
    print(f"\nOverall RMSE improvement: {overall_rmse_gain:.4f}")
    if n_cliff > 0:
        cliff_rmse_gain = m_base_c["rmse"] - m_res_c["rmse"]
        print(f"Cliff RMSE improvement: {cliff_rmse_gain:.4f}")

    # Per-molecule analysis: show examples
    print("\n" + "-" * 70)
    print("Example Predictions (Activity Cliff molecules):")
    print("-" * 70)
    if n_cliff > 0:
        cliff_idx = np.where(cliff_mask)[0][:10]  # Show up to 10
        print(f"{'Idx':>4s} | {'Actual':>7s} | {'Baseline':>8s} | "
              f"{'Residual':>8s} | {'Neighbor':>8s} | {'NB_sim':>6s}")
        print("-" * 65)
        for i in cliff_idx:
            print(f"{i:4d} | {y_test[i]:7.3f} | {pred_baseline[i]:8.3f} | "
                  f"{pred_residual[i]:8.3f} | {pred_neighbor[i]:8.3f} | "
                  f"{nb_sims_test[i]:6.3f}")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"""
Baseline Model: Standard FP → MLP
  - Overall RMSE: {m_base['rmse']:.4f}, R: {m_base['r']:.4f}

Residual Model: m(B) + g(B, FP(A), FP(B)-FP(A))
  - Overall RMSE: {m_res['rmse']:.4f}, R: {m_res['r']:.4f}
  - Uses MMP-like nearest neighbor from training set
  - Structural residual (FP(B)-FP(A)) as explicit feature

Key Insight:
  The residual model separates "expected activity for similar structures"
  from "activity deviation due to specific structural differences",
  which is exactly what activity cliffs are about.
""")


if __name__ == "__main__":
    main()

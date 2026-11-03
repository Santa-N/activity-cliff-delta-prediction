# Delta Prediction for Activity Cliff-Aware Molecular Property Prediction

Retrieval-augmented delta prediction approach for molecular activity prediction, with focus on activity cliff compounds.

## Key Idea

Instead of directly predicting a molecule's activity, we:
1. **Retrieve** structurally similar molecules (A) from a database
2. **Predict the change** (ΔY) from the reference molecule's known activity
3. **Final prediction**: `Ŷ(B) = Y(A) + ΔY(h_B, h_A, h_B - h_A)`

This is particularly effective for **activity cliffs** — structurally similar molecules with very different activities.

## Results on MoleculeACE Benchmark (30 datasets)

| Model | Overall RMSE | Cliff RMSE | Source |
|-------|:----------:|:--------:|:------:|
| REMOIM-Graphormer (SOTA) | 0.647 | 0.747 | published |
| SVM + ECFP | 0.662 | 0.750 | ours |
| XGBoost + ECFP | 0.694 | 0.775 | ours |
| RF + ECFP | 0.704 | 0.786 | ours |
| **GIN + Delta Pred (ours)** | **0.708** | **0.822** | ours |
| ECFP + Delta Pred (ours) | 0.720 | 0.831 | ours |
| D-MPNN | 0.750 | 0.829 | published |
| MLP + ECFP | 0.768 | 0.820 | ours |
| GIN Baseline | 0.808 | 0.869 | ours |
| GIN (published) | 0.815 | 0.901 | published |

## Experiments

### Core experiments
- `run_ecfp_delta.py` — **Main**: ECFP & GNN Delta Prediction vs SOTA baselines
- `run_advanced.py` — Advanced methods: RAG Attention, Extended Residual, Delta Pred, Combined

### Ablation studies
- `run_gnn_ablation.py` — GNN feature ablation + neighbor selection strategy
- `run_ablation.py` — MLP feature ablation + neighbor selection strategy

### Earlier experiments
- `run_moleculeace.py` — MLP baseline vs residual on MoleculeACE
- `run_gnn.py` — GNN baseline vs residual on MoleculeACE
- `run_sota_comparison.py` — Full SOTA comparison (RF, SVM, XGB, MLP, GIN, Delta)
- `run_experiment.py` / `run_experiment_v2.py` — Initial BACE dataset experiments
- `run_top1.py` / `run_gnn_top1.py` — Top-1 vs Top-3 neighbor comparison

## Key Findings

1. **Delta prediction (+10.5%)** outperforms direct prediction across all GNN architectures
2. **Structural residual (h_B - h_A)** in GNN embedding space is the most informative feature
3. **Top-3 weighted neighbor** averaging is more stable than Top-1
4. **Y(A) reference point** constrains prediction range, preventing large errors
5. **Simple models win**: Extended residuals and combined models overfit on this data scale

## Requirements

```
torch
torch_geometric
rdkit
scikit-learn
xgboost
MoleculeACE
```

## Related Work

- [SQRL](https://arxiv.org/abs/2501.09103) — Similarity-Quantized Relative Difference Learning
- [DeepDelta](https://doi.org/10.1186/s13321-023-00769-x) — Delta prediction for ADMET
- [REMO](https://arxiv.org/abs/2402.13779) — Reaction-based pre-training
- [MoleculeACE](https://doi.org/10.1021/acs.jcim.2c00613) — Activity cliff benchmark

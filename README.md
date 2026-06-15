# GNN-IDS-XAI — Graph Neural Network Intrusion Detection with Explainability

A reproducible, end-to-end study of network intrusion detection on the **UNSW-NB15** dataset, comparing traditional machine-learning models against Graph Neural Networks (GNNs) and adding model-agnostic and graph-native explainability (SHAP and GNNExplainer).

The whole pipeline lives in a single Jupyter notebook, `GNN-IDS-XAI_v3_english.ipynb`, organised into seven parts that run top to bottom.

> **Reproducibility is a first-class goal of this project.** Every stochastic step is seeded and every backend is forced into deterministic mode, so two people running the notebook on equivalent hardware and library versions get bit-for-bit identical models, predictions and explanations. The notebook ships with a self-test and a content fingerprint to prove it.

---

## What the notebook does

The dataset is framed two ways: **binary** (Normal vs. Attack) and **multiclass** (Normal + 9 attack categories: Analysis, Backdoor, DoS, Exploits, Fuzzers, Generic, Reconnaissance, Shellcode, Worms).

| Part | Topic | Summary |
|------|-------|---------|
| 1 | Traditional ML | Decision Tree, Random Forest and XGBoost with `GridSearchCV`; binary and multiclass; SHAP feature attribution |
| 2 | Optimised ML | Tuned XGBoost, LightGBM and Random Forest with early stopping and synthetic oversampling; native TreeSHAP |
| 3 | GraphSAGE GNN | Flows as nodes, edges from three temporal strategies; 9-config hyper-parameter search; inductive training; two-stage multiclass (GNN embeddings + MLP) |
| 4 | Explainability (XAI) | GNNExplainer on the GraphSAGE model: local subgraphs, per-node feature attribution, self-vs-contextual breakdown, global profiles |
| 5 | GCN / GAT | Same pipeline with GCN and GAT backbones for comparison |
| 6 | Ablation | Edge-type and feature-subset ablations |
| 7 | Telemetry | Low-telemetry strategy attribution (GNNExplainer lift vs. ablation recall-drop) |

### The graph

Network flows are nodes. Edges are built from three complementary temporal strategies and then deduplicated:

1. **Conversation chain** — consecutive flows of the same `(srcip, dstip)` pair.
2. **K-NN by source IP** — temporal neighbours sharing `srcip` (K=5).
3. **K-NN by destination IP** — temporal neighbours sharing `dstip` (K=3).

The split is **chronological** (70 / 15 / 15), so evaluation is genuinely inductive: the model is tested on flows that occur after everything it trained on.

---

## Results

Numbers below come from a full run on CPU (`torch 2.5.1`, `PyG 2.7.0`). They will be reproduced exactly on equivalent setups.

**Binary detection (Normal vs. Attack)** is strong and consistent across GNN backbones:

| Model | Accuracy | F1 | Recall |
|-------|----------|-----|--------|
| GraphSAGE | 0.991 | 0.993 | 1.000 |
| GCN | 0.988 | 0.990 | 1.000 |
| GAT | 0.987 | 0.990 | 0.999 |

**Multiclass** is harder, as expected on a heavily imbalanced 9-class problem; GraphSAGE leads the GNN family with Macro-F1 ≈ 0.41 / Weighted-F1 ≈ 0.71 (9 attack classes), while the best optimised traditional model (LightGBM) reaches Macro-F1 ≈ 0.66 on its own multiclass framing.

**Processing time** is reported separately for training and inference, with per-sample latency. Inference is cheap for every model (sub-millisecond to a few tenths of a millisecond per sample); GNN training is dominated by the hyper-parameter search and is markedly more expensive than the traditional models.

---

## Reproducibility

The notebook fixes all known sources of non-determinism:

- Global seeds for `random`, `numpy`, `torch` (CPU and CUDA), plus `PYTHONHASHSEED` and `CUBLAS_WORKSPACE_CONFIG`.
- `cudnn.deterministic = True`, `cudnn.benchmark = False`, and `torch.use_deterministic_algorithms(True, warn_only=True)` — this last one makes PyTorch Geometric's neighbour-aggregation *scatter* deterministic on GPU, which was the main reason a GNN produced different results on every run.
- A `set_seed()` helper called right before every stochastic block, so each cell is reproducible even when re-run on its own.
- Seeded `DataLoader` generators and per-node seeds inside the explainer loops.

**Two built-in checks let you verify the 100% yourself:**

1. **Determinism self-test** (end of Part 4) — re-runs the forward pass and GNNExplainer twice with the same seed and confirms, bit by bit, that the result is identical and matches the earlier cells.
2. **Reproducibility fingerprint** (final cell) — prints SHA-256 content hashes of every checkpoint, prediction and explainer mask, plus a single global fingerprint. Two collaborators compare fingerprints; if they match, their models, predictions and explanations are identical. Execution times are deliberately excluded from the hash.

For exact reproducibility across machines, use the **same library versions** and the **same kind of hardware** (CPU↔CPU or the same GPU). Across GPU generations, floats may differ in the last decimal even when the process is deterministic.

---

## Explainability

- **SHAP** (Parts 1–2) — feature attribution for the tree models, using XGBoost's native TreeSHAP for speed and exactness.
- **GNNExplainer** (Part 4) — for a target flow, it optimises a mask over the model (frozen) and returns a `[nodes × features]` importance matrix. The notebook summarises this matrix three complementary ways: a bar chart averaging all rows (whole-subgraph view), the subgraph labelling each node with its own most-influential feature, and a self-vs-contextual split of the target's row against its neighbours'.

The explainer uses the **same configuration (200 epochs) everywhere**, so the explanation of a given node is identical across every figure and every part of the notebook — verified by the self-test.

---

## Oversampling note (important)

Minority attack classes are oversampled with **KMeans-SMOTE**, applied consistently to both the GNN multiclass stage and the traditional models, keeping the same per-class target counts.

KMeans-SMOTE has a known limitation: it can fail to find valid clusters for some classes (it raises on a NaN density computation). When that happens the code **falls back to plain SMOTE and prints a loud warning**, so the method actually used is never ambiguous. In practice this fallback tends to trigger for the traditional models (Part 2), which oversample raw, sparse one-hot features, while the GNN stage (dense 128-dim embeddings) uses KMeans-SMOTE successfully. Check the cell output to see which sampler ran.

Because the oversampler determines the synthetic training data, **multiclass metrics differ from any earlier BorderlineSMOTE/RandomOverSampler runs** — this is expected, not a regression. Binary pipelines do not use oversampling and are unaffected.

---

## Getting started

### Requirements

- Python 3.10+
- `torch`, `torch_geometric`
- `scikit-learn`, `xgboost`, `lightgbm`, `imbalanced-learn`
- `shap`, `networkx`, `pandas`, `numpy`, `matplotlib`, `seaborn`

A GPU is optional; the notebook runs on CPU.

### Data

The first cell downloads the UNSW-NB15 files automatically from [Zenodo](https://zenodo.org/records/20528490) (public, no token needed) into `datasetUNSW/`. Citable DOI: `10.5281/zenodo.20528489`.

### Run

Open the notebook and use **Restart & Run All**. On the first run, training artefacts are cached under `checkpoints_*/` and timings under `model_timings_v3.json`; later runs reuse the checkpoints (much faster) instead of retraining. To regenerate everything from scratch, delete those folders.

```bash
jupyter notebook GNN-IDS-XAI_v3_english.ipynb
```

Part 7 (telemetry attribution) is the slowest section — it retrains GraphSAGE across many feature subsets and seeds, so allow time or skip it if you only need detection and explainability.

---

## Repository layout

```
.
├── GNN-IDS-XAI_v3_english.ipynb   # main notebook (Parts 1–7)
├── datasetUNSW/                   # auto-downloaded UNSW-NB15 CSVs
├── checkpoints_*/                 # cached models (created on first run)
├── model_timings_v3.json          # processing-time log (created on first run)
└── README.md
```

---


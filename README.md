# GNN-IDS-XAI — Graph Neural Network Models for Intrusion Detection Systems

Companion code for the paper:

> **Design, Implementation and Evaluation of Graph Neural Network Models for Intrusion Detection Systems**
> *[Authors]* — *[Journal, MDPI]*, *[Year]*. DOI: *[paper DOI]*

This repository contains a single, self-contained Jupyter notebook
(`GNN-IDS-XAI_v7.ipynb`) that reproduces every model, figure, table and
explainability result reported in the article, using the **UNSW-NB15**
dataset.

---

## 1. Overview

The notebook is organised as **two self-contained studies**, each following
the same internal flow — *train and evaluate the models first, then explain
them* — so that the (computationally expensive) explainability is run only
for the configuration whose performance is kept.

| Part | Models | Explainability |
|------|--------|----------------|
| **A — Traditional ML** | Decision Tree · Random Forest · XGBoost · LightGBM | **SHAP** |
| **B — Graph Neural Networks** | GraphSAGE · GCN · GAT · edge-type-aware GAT (RGAT) | **GNNExplainer** |

The two intrusion-detection problems addressed are:

- **Binary detection** — Normal vs. Attack.
- **Multiclass classification** — Normal + the UNSW-NB15 attack categories,
  solved with a **two-stage hierarchical scheme** (a binary GNN front-end,
  then an attack-type classifier trained on the learned node embeddings).

Beyond the core models, Part B includes an extended study with alternative
GNN backbones, structural ablations ("what does the graph buy?"),
limited-telemetry regimes, and robustness checks for the explanations.

---

## 2. Notebook structure

**Part A — Traditional Machine Learning** *(explainability: SHAP)*
- **A.1** Baseline models — all attack classes — training & evaluation
- **A.2** Baseline explainability — SHAP
- **A.3** Optimized models — aggressive preprocessing (`service='-'` dropped + `log1p`) — training & evaluation
- **A.4** Optimized explainability — SHAP

**Part B — Graph Neural Networks** *(explainability: GNNExplainer)*
- **B.1** GraphSAGE GNN-IDS — binary + two-stage multiclass — training & evaluation
- **B.2** GraphSAGE explainability — GNNExplainer (B.2.1 inline · B.2.2 standalone reproducible study)
- **B.3** Extended GNN study — backbones, ablations, telemetry & explainability robustness:
  - B.3.1 GCN & GAT backbones — head-to-head comparison
  - B.3.2 Refinements — credible attention + end-to-end multiclass
  - B.3.3 Edge-type-aware GAT (RGAT)
  - B.3.4 Structural ablation — what does the graph buy?
  - B.3.5 Limited-telemetry regime — when does the graph help?
  - B.3.6 Explainability validation under limited telemetry
  - B.3.7 Realistic limited telemetry — hiding whole sensor fields
  - B.3.8 GNNExplainer vs. edge-type ablation — head-to-head
- **B.4** GraphSAGE — aggressive preprocessing — training & evaluation
- **B.5** Aggressive-preprocessing explainability — GNNExplainer

**Appendix** — processing-time summary (training vs. inference, ms/sample) ·
reproducibility fingerprint.

> Sections marked with ⏭️ in the notebook are the heavy explainability blocks.
> They are fully independent of the model sections and can be skipped; their
> results are cached to disk.

---

## 3. Method summary

**Graph construction.** Each network flow becomes a node. Edges are built with
three temporal strategies that are later distinguishable for explanation:

1. **Conversation chain** — consecutive flows of the same `(srcip, dstip)` pair.
2. **K-NN per source IP** — temporal neighbours sharing the same `srcip` (`K_SRC`).
3. **K-NN per destination IP** — temporal neighbours sharing the same `dstip` (`K_DST`).

**Backbone (GraphSAGE).** Stacked `SAGEConv` layers with BatchNorm, ReLU and
dropout, a residual connection between hidden layers, and an MLP classification
head. An auxiliary `get_embeddings` method exposes the node representations used
by the two-stage multiclass scheme.

**Two-stage multiclass.** Stage 1 is the binary GraphSAGE. Stage 2 is an MLP
trained on the embeddings of the flows predicted as attacks, with class
imbalance handled by **KMeans-SMOTE** oversampling (train only). A combined
hierarchical evaluation reports metrics over the full taxonomy (Normal + attack
categories) for comparability with the literature.

**Evaluation protocol.** A **chronological 70/15/15** train/val/test split
(sorted by `Stime`) simulates an inductive deployment in which the model is
evaluated on *future* traffic.

---

## 4. Requirements

- **Python** ≥ 3.10
- A GPU is recommended for Part B (CUDA), but the notebook also runs on CPU.

Core libraries:

```
pandas
numpy
matplotlib
seaborn
networkx
scikit-learn
xgboost
lightgbm
imbalanced-learn
shap
torch                 # PyTorch
torch_geometric       # PyTorch Geometric (PyG)
```

> **Reproducibility note.** Exact numerical reproducibility between machines
> requires the **same versions** of `torch`, `torch_geometric`, `scikit-learn`,
> `xgboost` and `imbalanced-learn`, and the same kind of hardware
> (CPU↔CPU, or the same GPU). Across GPUs of different generations the last
> decimal of a float may differ even though the process is deterministic.
> Please pin and report the exact versions you used (see `requirements.txt`).

### Suggested installation

```bash
# 1) Create an environment
python -m venv .venv && source .venv/bin/activate      # or: conda create -n gnn-ids python=3.11

# 2) Install PyTorch (pick the build matching your CUDA / CPU from pytorch.org)
pip install torch

# 3) Install PyTorch Geometric (must match your torch / CUDA build — see PyG docs)
pip install torch_geometric

# 4) Install the remaining dependencies
pip install pandas numpy matplotlib seaborn networkx scikit-learn \
            xgboost lightgbm imbalanced-learn shap jupyter
```

---

## 5. Dataset

The notebook uses **UNSW-NB15** (Moustafa & Slay, 2015). Two views are used:

- The four raw CSVs (`UNSW-NB15_1..4.csv`) plus the feature description file.
- The official reduced train/test split (`UNSW_NB15_training-set.csv`,
  `UNSW_NB15_testing-set.csv`).

**Automatic download.** The first setup cell (`CELL 0`) downloads all files
from Zenodo into a local `datasetUNSW/` folder if they are not already present.
No account or token is required (the data is public).

- Zenodo record: `https://zenodo.org/records/20528490`
- Citable DOI: `10.5281/zenodo.20528489`

If you already have a local copy, place the CSVs under `datasetUNSW/` (or edit
the path at the top of the notebook) and the download step will be skipped.

---

## 6. How to run

1. Open the notebook:

   ```bash
   jupyter lab GNN-IDS-XAI_v7.ipynb
   ```

2. Run the setup cell to fetch the dataset.

3. From the menu, choose **Restart & Run All** for a clean, end-to-end
   reproduction. On a fresh machine the first run trains everything once
   (this can take a long time) and writes checkpoints; subsequent runs reuse
   them and are fast.

4. To run only part of the study, execute Part A or Part B independently — they
   are self-contained. Within each part, the model sections do not depend on the
   explainability (⏭️) sections, so the latter can be skipped.

---

## 7. Checkpoints & caching

Every heavy step follows a **load-if-exists, else compute-and-save** pattern,
so re-running a section is cheap and long searches/sweeps are resumable.

| Section | Checkpoint folder(s) | Cached content |
|---------|----------------------|----------------|
| A.1 — ML baseline | `checkpoints_ml_new/` | grid-searched best estimators |
| A.2 — ML baseline SHAP | `checkpoints_ml_new/*.pkl` | multiclass SHAP values + sample indices |
| A.3 — ML optimized | `checkpoints_ml_v4_new/` | trained DT / RF / XGB / LightGBM models |
| B.1 / B.2 — GraphSAGE + XAI | `checkpoints_new/` | best hyper-parameters, final graphs, trained GNN, global explainer aggregation |
| B.3 — extended GNN | `checkpoints_gcn/`, `checkpoints_gat/`, `checkpoints_*_e2e/`, `checkpoints_rgat/`, `checkpoints_ablation/`, `checkpoints_rgat_t/` | per-backbone models, ablation ladder, telemetry sweeps |
| B.4 / B.5 — GNN aggressive | `checkpoints_gnn_aggr/` | two-stage GNN retrained on aggressively-cleaned data + per-category GNNExplainer features |
| Appendix | `model_timings_v3.json` | per-model training / inference timings |

**Force a clean retrain:** delete the relevant `checkpoints_*` folder (and
`model_timings_v3.json` if you also want to re-measure times) and re-run.
Sharing checkpoints between collaborators guarantees identical results without
retraining.

---

## 8. Reproducibility

The notebook is engineered so that two collaborators obtain the same results in
the GNN and explainability parts:

- Every imports cell fixes **all** randomness sources (`random`, `numpy`,
  `torch` CPU/CUDA), enables deterministic kernels
  (`cudnn.deterministic=True`, `cudnn.benchmark=False`,
  `torch.use_deterministic_algorithms(...)`) and sets
  `CUBLAS_WORKSPACE_CONFIG`. This removes the non-determinism of PyG's scatter
  aggregation on GPU.
- A `set_seed(SEED)` helper is called right before every stochastic block
  (model init, training, SHAP sampling, each GNNExplainer call), so each cell is
  reproducible even when re-run in isolation or out of order. Default `SEED = 42`.
- A single, unified GNNExplainer configuration (200 epochs) is used throughout,
  so the explanation of a given node is bit-identical across sections.

Two built-in verification tools:

1. **Determinism self-test** (end of the B.2.2 study) — re-runs the forward
   pass and GNNExplainer twice with the same seed and checks bit-for-bit
   equality.
2. **Reproducibility fingerprint** (final cell) — prints content hashes of
   checkpoints, predictions and masks. Matching versions + equivalent hardware
   should yield the same global fingerprint.

> On reading the explanations: `node_mask` is a `[nodes × features]` matrix, so
> different figures (subgraph-mean bar chart, per-node argmax labels,
> self-vs-context split) are different summaries of the **same** explanation.
> Highlighting different features in different views is expected and does not
> indicate a reproducibility failure.

---

## 9. File layout

```
.
├── GNN-IDS-XAI_v7.ipynb     # the complete study (this notebook)
├── README.md                # this file
├── requirements.txt         # pinned dependency versions (recommended to add)
├── datasetUNSW/             # created on first run (downloaded from Zenodo)
├── checkpoints_*/           # created on first run (cached models / results)
└── model_timings_v3.json    # created on first run (timing log)
```

---

## 10. Citation

If you use this code or build on it, please cite the paper and the dataset.

**Paper:**

```bibtex
@article{[citekey],
  title   = {Design, Implementation and Evaluation of Graph Neural Network Models for Intrusion Detection Systems},
  author  = {[Authors]},
  journal = {[MDPI Journal]},
  year    = {[Year]},
  doi     = {[paper DOI]}
}
```

**Dataset (this Zenodo deposit):**

```bibtex
@dataset{unsw_nb15_zenodo,
  title     = {UNSW-NB15},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.20528489},
  url       = {https://zenodo.org/records/20528490}
}
```

Please also cite the original UNSW-NB15 reference (Moustafa, N.; Slay, J.
*UNSW-NB15: a comprehensive data set for network intrusion detection systems*,
MilCIS 2015).

---

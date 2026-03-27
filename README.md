# Fashion-MNIST: Classifier Comparison (MLP, SVM, RVM)

Classifying the Fashion-MNIST dataset (10 clothing categories, 28×28 grayscale images) using three families of models: Multi-Layer Perceptrons, Support Vector Machines, and Relevance Vector Machines.

---

## Dataset

| Split      | Size   | Notes                                 |
|------------|--------|---------------------------------------|
| Train      | 54,000 | Random split from original 60k train  |
| Validation | 6,000  | Held out for tuning (seed=42)         |
| Test       | 10,000 | Official FashionMNIST test set        |

Preprocessing: pixel values normalised with mean=0.2860, std=0.3530.

---

## Model 1: MLP (`build_mlp.py`)

A fully-connected network built with PyTorch. Each case isolates one variable from the previous.

### Cases

| Case         | Architecture    | Activation | Dropout | Batch Norm | Optimizer | LR    |
|--------------|-----------------|------------|---------|------------|-----------|-------|
| `mlp_base`   | [128]           | ReLU       | 0.0     | No         | Adam      | 0.001 |
| `mlp_case_1` | [256]           | ReLU       | 0.0     | No         | Adam      | 0.001 |
| `mlp_case_2` | [256, 128]      | ReLU       | 0.0     | No         | Adam      | 0.001 |
| `mlp_case_3` | [256, 128]      | ReLU       | 0.3     | No         | Adam      | 0.001 |
| `mlp_case_4` | [256, 128]      | ReLU       | 0.3     | Yes        | Adam      | 0.001 |
| `mlp_case_5` | [256, 128]      | GELU       | 0.3     | No         | Adam      | 0.001 |
| `mlp_case_6` | [256, 128]      | ReLU       | 0.3     | No         | SGD       | 0.01  |
| `mlp_best`   | [512, 256, 128] | GELU       | 0.3     | Yes        | AdamW     | 0.001 |

### Results

| Case         | Params    | Test Acc | Macro F1 | Train Time |
|--------------|-----------|----------|----------|------------|
| `mlp_base`   | 101,770   | 89.20%   | 0.8923   | 171.6s     |
| `mlp_case_1` | 203,530   | 89.51%   | 0.8952   | 169.2s     |
| `mlp_case_2` | 235,146   | 89.45%   | 0.8951   | 258.7s     |
| `mlp_case_3` | 235,146   | 89.45%   | 0.8943   | 166.2s     |
| `mlp_case_4` | 235,914   | 89.76%   | 0.8973   | 488.3s     |
| `mlp_case_5` | 235,146   | 89.59%   | 0.8954   | 205.6s     |
| `mlp_case_6` | 235,146   | 85.91%   | 0.8584   | 149.6s     |
| `mlp_best`   | 569,226   | **90.14%** | **0.9012** | 203.3s |

### Key Findings — MLP

- Widening from [128] to [256] gives a small gain; depth ([256, 128]) adds minimal benefit on its own.
- Dropout (0.3) does not hurt accuracy and slightly improves val generalisation.
- Batch normalisation + dropout together (case 4) is more effective than either alone.
- GELU outperforms ReLU slightly on this dataset.
- SGD (case 6) underperforms Adam-family optimisers significantly (−3.8pp) even with a tuned LR.
- Best MLP combines all improvements: deeper [512, 256, 128] + GELU + dropout + BN + AdamW → **90.14%**.

---

## Model 2: SVM (`build_svm.py`)

scikit-learn `SVC` with one-vs-rest multiclass. Data is flattened to 784-dim vectors; SVM cases use a fixed 640-sample subset (10 batches × 64) unless stated otherwise.

### Cases

| Case            | Kernel | C     | Gamma | PCA | Train Size |
|-----------------|--------|-------|-------|-----|------------|
| `baseline_svm`  | RBF    | 1.0   | scale | —   | 640        |
| `poly_kernel`   | Poly   | 1.0   | scale | —   | 640        |
| `tune_C=0.1`    | RBF    | 0.1   | scale | —   | 640        |
| `tune_C=1`      | RBF    | 1.0   | scale | —   | 640        |
| `tune_C=10`     | RBF    | 10.0  | scale | —   | 640        |
| `tune_C=100`    | RBF    | 100.0 | scale | —   | 640        |
| `tune_gamma=scale` | RBF | 1.0  | scale | —   | 640        |
| `tune_gamma=auto`  | RBF | 1.0  | auto  | —   | 640        |
| `tune_gamma=0.001` | RBF | 1.0  | 0.001 | —  | 640        |
| `tune_gamma=0.01`  | RBF | 1.0  | 0.01  | —  | 640        |
| `more_data=5`   | RBF    | 1.0   | scale | —   | 320        |
| `more_data=20`  | RBF    | 1.0   | scale | —   | 1,280      |
| `more_data=50`  | RBF    | 1.0   | scale | —   | 3,200      |
| `pca_n=50`      | RBF    | 1.0   | scale | 50  | 640        |
| `pca_n=100`     | RBF    | 1.0   | scale | 100 | 640        |
| `best_svm`      | RBF    | 10.0  | scale | —   | 3,200      |

### Results

| Case               | Train Size | Support Vecs | Test Acc | Macro F1 | Train Time |
|--------------------|------------|--------------|----------|----------|------------|
| `baseline_svm`     | 640        | 493          | 76.56%   | 0.7765   | <1s        |
| `poly_kernel`      | 640        | 486          | 68.95%   | 0.7175   | <1s        |
| `tune_C=0.1`       | 640        | 629          | 59.77%   | 0.5779   | <1s        |
| `tune_C=1`         | 640        | 511          | 76.95%   | 0.7665   | <1s        |
| `tune_C=10`        | 640        | 511          | 78.71%   | 0.7960   | <1s        |
| `tune_C=100`       | 640        | 500          | 80.86%   | 0.8159   | <1s        |
| `tune_gamma=scale` | 640        | 494          | 79.88%   | 0.7974   | <1s        |
| `tune_gamma=auto`  | 640        | 496          | 77.34%   | 0.7782   | <1s        |
| `tune_gamma=0.001` | 640        | 500          | 78.71%   | 0.7905   | <1s        |
| `tune_gamma=0.01`  | 640        | 622          | 70.90%   | 0.7129   | <1s        |
| `more_data=5`      | 320        | 278          | 74.22%   | 0.7411   | <1s        |
| `more_data=20`     | 1,280      | 881          | 81.64%   | 0.8214   | <1s        |
| `more_data=50`     | 3,200      | 1,868        | 84.38%   | 0.8474   | 0.5s       |
| `pca_n=50`         | 640        | 499          | 79.69%   | 0.8054   | <1s        |
| `pca_n=100`        | 640        | 492          | 78.12%   | 0.7867   | <1s        |
| `best_svm`         | 3,200      | 1,791        | **87.11%** | **0.8780** | 0.5s   |

### Key Findings — SVM

- RBF kernel is clearly superior to Polynomial for this task (76.56% vs 68.95%).
- C is the most impactful hyperparameter: accuracy rises monotonically from 59.77% (C=0.1) to 80.86% (C=100) on the same data.
- Gamma has a narrower effect; `scale` and `0.001` perform similarly, while `0.01` degrades accuracy (too many support vectors: 622).
- **Training size dominates everything**: scaling from 320 to 3,200 samples gives +10pp regardless of kernel tuning.
- PCA compression (50 or 100 components) recovers most accuracy vs. raw 784 features with a minor loss (≤1pp), useful for speed.
- Best SVM: RBF + C=10 + 3,200 samples → **87.11%**, still ~3pp below the best MLP.

---

## Model 3: RVM (`build_rvm.py`)

Relevance Vector Classifier (`skrvm.RVC`) — a Bayesian sparse kernel method. Produces probabilistic outputs and typically uses far fewer relevance vectors than SVM support vectors.

### Cases

| Case            | Kernel | n_iter | tol   | Train Size |
|-----------------|--------|--------|-------|------------|
| `baseline_rvm`  | RBF    | 3000   | 1e-3  | varies     |
| `linear_rvm`    | Linear | 3000   | 1e-3  | varies     |
| `poly_rvm`      | Poly   | 3000   | 1e-3  | varies     |
| `tune_n_iter`   | RBF    | 100–3000 | 1e-3 | varies    |
| `tune_tol`      | RBF    | 3000   | 1e-2 – 1e-4 | varies |
| `data_size`     | RBF    | 3000   | 1e-3  | 2–10 batches |
| `svm_vs_rvm`    | RBF    | 3000   | 1e-3  | matched    |
| `best_rvm`      | best   | best   | best  | max        |

> Results CSV pending — run `build_rvm.py` to generate `results/rvm_experiments.csv`.

### Expected Key Findings — RVM

- RVM uses significantly fewer relevance vectors than SVM support vectors, making it more interpretable and memory-efficient at inference time.
- Training time is substantially higher than SVM due to the Bayesian update iterations.
- RBF kernel expected to outperform linear/poly (consistent with SVM findings).
- Increasing `n_iter` improves convergence but with diminishing returns above 1000 iterations.
- Data size still the dominant factor in accuracy, same as SVM.

---

## Cross-Model Comparison

| Model      | Best Test Acc | Macro F1 | Notes                              |
|------------|---------------|----------|------------------------------------|
| MLP (best) | **90.14%**    | **0.9012** | [512,256,128] + GELU + BN + AdamW |
| SVM (best) | 87.11%        | 0.8780   | RBF, C=10, 3,200 samples           |
| RVM (best) | TBD           | TBD      | Results pending                    |

MLP leads by ~3pp over SVM at comparable data scales, but SVM trains orders of magnitude faster on small subsets. RVM offers sparsity and probabilistic outputs at the cost of training time.

---

## Project Structure

```
Fashion-MNIST/
├── data.py            # Dataset loading & splits (train/val/test loaders)
├── build_mlp.py       # MLP builder, trainer, and 8-case experiment runner
├── build_svm.py       # SVM builder and 8-case experiment runner
├── build_rvm.py       # RVM builder and 8-case experiment runner
├── data/              # Auto-downloaded FashionMNIST raw files
└── results/
    ├── mlp_experiments.csv
    ├── svm_experiments.csv
    └── rvm_experiments.csv   
```

## Running

```bash
python build_mlp.py   # trains all 8 MLP cases, saves results/mlp_experiments.csv
python build_svm.py   # trains all 8 SVM cases, saves results/svm_experiments.csv
python build_rvm.py   # trains all 8 RVM cases, saves results/rvm_experiments.csv
```

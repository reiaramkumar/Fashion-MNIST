# ...................... RVM .......................

# Case 1: baseline_rvm   → RBF, defaults
# Case 2: linear_rvm     → kernel: rbf → linear
# Case 3: poly_rvm       → kernel: poly degree=3
# Case 4: tune_n_iter    → n_iter: 100, 300, 1000, 3000
# Case 5: tune_tol       → tol: 1e-2, 1e-3, 1e-4
# Case 6: data_size      → num_batches: 2→5→10
# Case 7: svm_vs_rvm     → Direct comparison at same data
# Case 8: best_rvm       → best kernel + n_iter + max data

import time, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report 

from skrvm import RVC

from data import train_loader, val_loader, test_loader

_rvm_results_log = []

# HELPER FUNCTION: prepare_data
# Extract data form the loaders and prepare for Sklearn (Numpy & Flatten) --> preperation for SVM & RVM
def prepare_data(loader, num_batches=10):
    x_list, y_list = [], []
    for i, (imgs, labels) in enumerate(loader):
        if i >= num_batches: break
        # Flatten: [Batch, 1, 28, 28] -> [Batch, 784]
        x_list.append(imgs.view(imgs.size(0), -1).numpy())
        y_list.append(labels.numpy())
        print(f"  [batch {i+1}/{num_batches} loaded]")
    return np.concatenate(x_list), np.concatenate(y_list)

# .... RVM BUILDER ....

def _build_rvm(
    kernel      =   "rbf",
    degree      =   3,
    coef1       =   None,
    coef0       =   0.0,
    n_iter      =   3000,
    tol         =   1e-3,
    alpha       =   1e-6,
    threshold_alpha=1e9,
    beta        =   1.0e-6,
    beta_fixed  =   False,
    bias_used   =   True,
    verbose     =   False,
):
    model = RVC(
        kernel          = kernel,
        degree          = degree,
        n_iter          = n_iter,
        tol             = tol,
        threshold_alpha = threshold_alpha,
    )
    return model


def _run_rvm(
    name            = 'rvm_',
    kernel          = 'rbf',
    degree          = 3,
    n_iter          = 3000,
    tol             = 1e-3,
    num_batches     = 10,
):
    print(f"\n{'.'*50}")
    print(f"  {name}")
    print(f"{'.'*50}")

    # Load data
    X_tr, y_tr = prepare_data(train_loader, num_batches=num_batches)
    X_te, y_te = prepare_data(test_loader,  num_batches=2)

    
    # Build and train
    model = _build_rvm(kernel=kernel, n_iter=n_iter, tol=tol, degree=degree)
    t0 = time.time()
    model.fit(X_tr, y_tr)
    train_time = time.time() - t0
    estimators   = model.multi_.estimators_
    rv_counts    = [len(est.relevance_) for est in estimators]
    n_relevance  = sum(rv_counts)

    if any(c == 0 for c in rv_counts):
        print(f"  [WARNING] {name}: one or more classes have 0 relevance vectors — skipping evaluation")
        return None

    # Evaluate
    y_pred      = model.predict(X_te)
    test_acc    = model.score(X_te, y_te)
    report      = classification_report(y_te, y_pred, output_dict=True, zero_division=0)
    macro_f1    = report['macro avg']['f1-score']


    results = dict(
        name            = name,
        kernel          = kernel,
        n_iter          = n_iter,
        tol             = tol,
        degree          = degree if kernel == 'poly' else '-',
        num_batches     = num_batches,
        train_size      = len(X_tr),
        test_acc        = round(test_acc * 100, 2),
        macro_f1        = round(macro_f1, 4),
        train_time      = round(train_time, 1),
        y_pred          = y_pred,
        y_true          = y_te,
        model           = model,
    )
    _rvm_results_log.append(results)

    print(f"\n{'─'*50}")
    print(f"  {name}")
    print(f"  kernel={kernel}" +
          (f"  degree={degree}" if kernel == 'poly' else '') +
           f"  n_iter={n_iter}  tol={tol}")

    print(f"  test accuracy: {test_acc*100:.2f}%   macro F1: {macro_f1:.4f}")
    print(f"  training time: {train_time:.1f}s")
    print(f"{'─'*50}\n")
    save_rvm('results')

    return results


def save_rvm(save_dir=None):
    rows = [{k: v for k, v in r.items()
             if k not in ('y_pred', 'y_true', 'model', 'pca')}
            for r in _rvm_results_log]
    df = pd.DataFrame(rows)
    print(df[['name', 'kernel', 'n_iter', 'tol', 'degree', 'num_batches',
              'train_size', 'n_relevance', 'test_acc', 'macro_f1', 'train_time']
             ].to_string(index=False))
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        df.to_csv(os.path.join(save_dir, 'rvm_experiments.csv'), index=False)
        print(f"[saved] {os.path.join(save_dir, 'rvm_experiments.csv')}")
    return df



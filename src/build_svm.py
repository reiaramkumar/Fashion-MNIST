# ...................... SVM .......................

# Case 1: baseline_svm   → RBF, C=1, default gamma
# Case 2: linear_kernel  → kernel: rbf → linear
# Case 3: poly_kernel    → kernel: rbf → poly degree=3
# Case 4: tune_C         → C: 0.1, 1, 10, 100
# Case 5: tune_gamma     → gamma: scale, auto, 0.001, 0.01
# Case 6: more_data      → num_batches: 5→20→50
# Case 7: pca_features   → PCA 784→50/100 before SVM
# Case 8: best_svm       → best C + gamma + max data


# class SVC(sklearn.svm._base.BaseSVC)
#  SVC(*, C=1.0, kernel='rbf', degree=3, gamma='scale',
#      coef0=0.0, shrinking=True, probability=False, 
#      tol=0.001, cache_size=200, class_weight=None, 
#      verbose=False, max_iter=-1, decision_function_shape='ovr',
#      break_ties=False, random_state=None)
import time, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report
from data import train_loader, val_loader, test_loader

_svm_results_log = []

# HELPER FUNCTION: prepare_data
# Extract data form the loaders and prepare for Sklearn (Numpy & Flatten) --> preperation for SVM & RVM
def prepare_data(loader, num_batches=10):
    x_list, y_list = [], []
    for i, (imgs, labels) in enumerate(loader):
        if i >= num_batches: break
        # Flatten: [Batch, 1, 28, 28] -> [Batch, 784]
        x_list.append(imgs.view(imgs.size(0), -1).numpy())
        y_list.append(labels.numpy())
    return np.concatenate(x_list), np.concatenate(y_list)

# .... SVM BUILDER ....

def _build_svm(
    kernel          = 'rbf',
    C               = 1.0,
    gamma           = 'scale',
    degree          = 3,        # only used when kernel='poly'
    pca_components  = None,     # None = no PCA, int = reduce to n components
):
    model = SVC(
        kernel  = kernel,
        C       = C,
        gamma   = gamma,
        degree  = degree,
    )
    return model, pca_components


def _run_svm(
    name            = 'svm_',
    kernel          = 'rbf',
    C               = 1.0,
    gamma           = 'scale',
    degree          = 3,
    pca_components  = None,
    num_batches     = 10,
):
    print(f"\n{'.'*50}")
    print(f"  {name}")
    print(f"{'.'*50}")

    # Load data
    X_tr, y_tr = prepare_data(train_loader, num_batches=num_batches)
    X_te, y_te = prepare_data(test_loader,  num_batches=2)

    # Optional PCA — fit on train only to avoid leakage
    pca = None
    if pca_components is not None:
        pca = PCA(n_components=pca_components, random_state=42)
        X_tr = pca.fit_transform(X_tr)
        X_te = pca.transform(X_te)

    # Build and train
    model, _ = _build_svm(kernel=kernel, C=C, gamma=gamma,
                           degree=degree, pca_components=pca_components)
    t0 = time.time()
    model.fit(X_tr, y_tr)
    train_time = time.time() - t0

    # Evaluate
    y_pred      = model.predict(X_te)
    test_acc    = model.score(X_te, y_te)
    report      = classification_report(y_te, y_pred, output_dict=True, zero_division=0)
    macro_f1    = report['macro avg']['f1-score']
    n_support   = model.n_support_.sum()

    results = dict(
        name            = name,
        kernel          = kernel,
        C               = C,
        gamma           = gamma,
        degree          = degree if kernel == 'poly' else '-',
        pca_components  = pca_components if pca_components else '-',
        num_batches     = num_batches,
        train_size      = len(X_tr),
        n_support       = n_support,
        test_acc        = round(test_acc * 100, 2),
        macro_f1        = round(macro_f1, 4),
        train_time      = round(train_time, 1),
        y_pred          = y_pred,
        y_true          = y_te,
        model           = model,
        pca             = pca,
    )
    _svm_results_log.append(results)

    print(f"\n{'─'*50}")
    print(f"  {name}")
    print(f"  kernel={kernel}  C={C}  gamma={gamma}" +
          (f"  degree={degree}" if kernel == 'poly' else '') +
          (f"  PCA={pca_components}" if pca_components else ''))
    print(f"  train_size   : {len(X_tr):,}  support_vectors={n_support}")
    print(f"  test accuracy: {test_acc*100:.2f}%   macro F1: {macro_f1:.4f}")
    print(f"  training time: {train_time:.1f}s")
    print(f"{'─'*50}\n")
    save_svm('results')

    return results


def save_svm(save_dir=None):
    rows = [{k: v for k, v in r.items()
             if k not in ('y_pred', 'y_true', 'model', 'pca')}
            for r in _svm_results_log]
    df = pd.DataFrame(rows)
    print(df[['name', 'kernel', 'C', 'gamma', 'pca_components',
              'train_size', 'n_support', 'test_acc', 'macro_f1', 'train_time']
             ].to_string(index=False))
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        df.to_csv(os.path.join(save_dir, 'svm_experiments.csv'), index=False)
        print(f"[saved] {os.path.join(save_dir, 'svm_experiments.csv')}")
    return df


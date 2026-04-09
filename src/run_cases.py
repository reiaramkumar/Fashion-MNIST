from build_mlp import _run_mlp, _results_log
from build_svm import _run_svm, _svm_results_log
from build_rvm import _run_rvm, _rvm_results_log


def svm_run_cases():
    # Case 1: baseline
    _run_svm('baseline_svm',  kernel='rbf',    C=1.0,  gamma='scale', num_batches=10)


    # Case 3: poly kernel
    _run_svm('poly_kernel',   kernel='poly',   C=1.0,  gamma='scale', degree=3, num_batches=10)

    # Case 4: tune C
    for C in [0.1, 1, 10, 100]:
        _run_svm(f'tune_C={C}',   kernel='rbf', C=C,   gamma='scale', num_batches=10)

    # Case 5: tune gamma
    for g in ['scale', 'auto', 0.001, 0.01]:
        _run_svm(f'tune_gamma={g}', kernel='rbf', C=1.0, gamma=g,     num_batches=10)

    # Case 6: more data
    for nb in [5, 20, 50]:
        _run_svm(f'more_data={nb}', kernel='rbf', C=1.0, gamma='scale', num_batches=nb)

    # Case 7: PCA features
    for n in [50, 100]:
        _run_svm(f'pca_n={n}',    kernel='rbf', C=1.0, gamma='scale', pca_components=n, num_batches=10)

    # # Case 8: best — update after seeing cases 4 & 5
    _run_svm('best_svm', kernel='rbf', C=10, gamma='scale', num_batches=50)


def mlp_run_cases():
    print("*" * 60)
    print("  MLP EXPERIMENTS")
    print("*" * 60)

    # Case 1: baseline model
    _run_mlp('mlp_base', hidden_size=[128], activation='relu', dropout=0.0, epochs=20)

    # Case 2: larger width
    _run_mlp('mlp_case_1', hidden_size=[256], activation='relu', dropout=0.0, epochs=20)

    # Case 3: deeper network
    _run_mlp('mlp_case_2', hidden_size=[256, 128], activation='relu', dropout=0.0, epochs=20)

    # Case 4: dropout regularization
    _run_mlp('mlp_case_3', hidden_size=[256, 128], activation='relu', dropout=0.3, epochs=20)

    # Case 5: batch normalization
    _run_mlp('mlp_case_4', hidden_size=[256, 128], activation='relu', dropout=0.3, batch_norm=True, epochs=20)

    # Case 6: GELU activation
    _run_mlp('mlp_case_5', hidden_size=[256, 128], activation='gelu', dropout=0.3, epochs=20)

    # Case 7: SGD optimizer
    _run_mlp('mlp_case_6', hidden_size=[256, 128], activation='relu', dropout=0.3, optimizer='sgd', lr=0.01, epochs=20)

    # Case 8: best configuration
    _run_mlp('mlp_best', hidden_size=[512, 256, 128], activation='gelu', dropout=0.3, batch_norm=True,
             optimizer='adamw', weight_decay=1e-4, epochs=20)


def rvm_run_cases():
        # Case 1: baseline
    _run_rvm('baseline_rvm',  kernel='rbf', degree = 3, n_iter= 3000, tol=0.001, num_batches=10)

    # Case 2: linear kernel
    _run_rvm('linear_kernel', kernel='linear', degree=3, n_iter=3000, tol=0.001, num_batches=10)

    # Case 3: poly kernel
    _run_rvm('poly_kernel',   kernel='poly',   degree=3, n_iter=3000, tol=0.001, num_batches=10)

    # Case 4: tune n_iter
    for n_it in [100, 300, 1000, 3000]:
        _run_rvm(f'tune_iter={n_it}',   kernel='rbf', degree=3, n_iter=n_it, tol=0.001, num_batches=10)

    # Case 5: tune tol
    for t in [5e-4, 1e-3, 1e-2, 1e-1]:
        _run_rvm(f'tune_tol={t}', kernel='rbf', degree=3, n_iter=3000, tol=t, num_batches=10)

    # Case 6: more data
    for nb in [5, 10, 15]:
        _run_rvm(f'more_data={nb}', kernel='rbf', degree=3, n_iter=3000, tol=t, num_batches=nb)

    # Case 7: best — update after seeing cases 4 & 5
    _run_rvm('best_rvm',      kernel='rbf',    degree = 3, n_iter= 3000, tol=0.001, num_batches=20)

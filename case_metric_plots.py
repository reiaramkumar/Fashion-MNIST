# .... IMPORTS  ....
from matplotlib import pyplot as plt
import matplotlib as mpl
import os
from sklearn.metrics import classification_report

mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['axes.titlesize'] = 13
mpl.rcParams['axes.labelsize'] = 12
mpl.rcParams['xtick.labelsize'] = 11
mpl.rcParams['ytick.labelsize'] = 11
mpl.rcParams['legend.fontsize'] = 10

# .... PLOT CURVES ....
linestyles = ['-', '--', '-.', ':', '-', '--', '-.', ':']
_COLORS = ['#1f4e79', '#c00000', '#375623', '#7030a0',
           '#833c00', '#215868', '#843c0c', '#4f6228']


def _style_ax(ax):
    """Clean formal axes style."""
    ax.set_facecolor('white')
    for spine in ax.spines.values():
        spine.set_edgecolor('#333333')
        spine.set_linewidth(0.9)
    ax.tick_params(colors='black', length=4, width=0.8)
    ax.xaxis.label.set_color('black')
    ax.yaxis.label.set_color('black')
    ax.title.set_color('black')
    ax.grid(True, color='#cccccc', alpha=0.6, linewidth=0.6, linestyle='--')
    ax.set_axisbelow(True)


# ──────────────────────────────────────────────────────────────────────────────
# PLOT CURVES  (MLP only — loss & accuracy vs epoch)
# ──────────────────────────────────────────────────────────────────────────────
def _plot_curves(model, result):

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), facecolor='white')
    fig.patch.set_facecolor('white')

    for ax in (ax1, ax2):
        _style_ax(ax)

    for i, r in enumerate(result):
        epochs_range = range(1, len(r['history']['train_loss']) + 1)
        ax1.plot(epochs_range, r['history']['train_loss'],
                 color=_COLORS[i % len(_COLORS)], linestyle=linestyles[i % len(linestyles)],
                 linewidth=1.8, label=r['name'])
        ax2.plot(epochs_range, [v * 100 for v in r['history']['val_acc']],
                 color=_COLORS[i % len(_COLORS)], linestyle=linestyles[i % len(linestyles)],
                 linewidth=1.8, label=r['name'])

    ax1.set_title(f'{model.upper()} — Training Loss per Epoch', pad=10)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Cross-Entropy Loss')
    ax1.set_xlim(left=1)

    ax2.set_title(f'{model.upper()} — Validation Accuracy per Epoch', pad=10)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_xlim(left=1)

    legend_kw = dict(facecolor='white', edgecolor='#333333',
                     framealpha=0.9, labelcolor='black', ncol=2)
    ax1.legend(loc='upper right', **legend_kw)
    ax2.legend(loc='lower right', **legend_kw)

    fig.suptitle(f'{model.upper()} — Training Loss and Validation Accuracy',
                 fontsize=14, color='black', y=1.02)
    plt.tight_layout()

    save_dir = model
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f'{model}_case_metrics.png'),
                dpi=300, facecolor='white', bbox_inches='tight')
    plt.show()


# ──────────────────────────────────────────────────────────────────────────────
# COMPARE RESULTS  (bar charts: accuracy, time, per-class F1, macro F1)
# ──────────────────────────────────────────────────────────────────────────────
def _compare_results(model, result):

    n = len(result)
    bar_w = max(10, n * 1.8)   # wider for SVM (15 cases)
    bar_h = 6                   # increased height for all models

    # ── Accuracy ────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(bar_w, bar_h), facecolor='white')
    _style_ax(ax)

    names    = [r['name'] for r in result]
    accuracy = [r['test_acc'] for r in result]
    bars = ax.bar(names, accuracy, color='#1f4e79', width=0.55, zorder=3)
    bars[accuracy.index(max(accuracy))].set_color('#375623')

    for bar, acc in zip(bars, accuracy):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.25,
                f'{acc:.1f}%', ha='center', va='bottom', fontsize=10)

    plt.xticks(rotation=25, ha='right')
    ax.set_title(f'{model.upper()} — Test Accuracy per Case', pad=10)
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_ylim(max(0, min(accuracy) - 4), min(100, max(accuracy) + 4))

    plt.tight_layout()
    os.makedirs(model, exist_ok=True)
    plt.savefig(os.path.join(model, f'{model}_accuracy_metrics.png'),
                dpi=300, facecolor='white', bbox_inches='tight')
    plt.show()

    # ── Training Time ────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(bar_w, bar_h), facecolor='white')
    _style_ax(ax)

    times = [r['train_time'] for r in result]
    bars = ax.bar(names, times, color='#1f4e79', width=0.55, zorder=3)
    bars[times.index(min(times))].set_color('#375623')

    for bar, t in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(times) * 0.01,
                f'{t:.1f}s', ha='center', va='bottom', fontsize=10)

    plt.xticks(rotation=25, ha='right')
    ax.set_title(f'{model.upper()} — Training Time per Case', pad=10)
    ax.set_ylabel('Training Time (s)')
    ax.set_ylim(0, max(times) * 1.18)

    plt.tight_layout()
    plt.savefig(os.path.join(model, f'{model}_train_time_metrics.png'),
                dpi=300, facecolor='white', bbox_inches='tight')
    plt.show()

    # ── Per-Class F1 ─────────────────────────────────────────────────────────
    class_names = ['T-Shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot']

    cmap_colors = (list(plt.cm.tab20.colors) if n > 8
                   else _COLORS)

    fig, ax = plt.subplots(figsize=(15, 5.5), facecolor='white')
    _style_ax(ax)

    all_f1 = []
    for i, r in enumerate(result):
        report    = classification_report(r['y_true'], r['y_pred'],
                                          output_dict=True, zero_division=0)
        f1_scores = [report[str(c)]['f1-score'] for c in range(len(class_names))]
        all_f1.append(f1_scores)
        ax.plot(class_names, f1_scores,
                color=cmap_colors[i % len(cmap_colors)],
                linestyle=linestyles[i % len(linestyles)],
                linewidth=1.6, marker='o', markersize=4, label=r['name'])

    ax.set_ylim(max(0, min(min(f) for f in all_f1) - 0.05), 1.05)
    ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1),
              facecolor='white', edgecolor='#333333',
              labelcolor='black', framealpha=0.9, borderaxespad=0)
    ax.set_ylabel('F1 Score')
    ax.set_title(f'{model.upper()} — Per-Class F1 Score', pad=10)
    plt.xticks(rotation=30, ha='right')

    plt.tight_layout()
    plt.savefig(os.path.join(model, f'{model}_F1_score_metrics.png'),
                dpi=300, facecolor='white', bbox_inches='tight')
    plt.show()

    # ── Macro F1 ─────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(bar_w, bar_h), facecolor='white')
    _style_ax(ax)

    macro_f1 = [
        classification_report(r['y_true'], r['y_pred'],
                               output_dict=True, zero_division=0)['macro avg']['f1-score']
        for r in result
    ]
    bars = ax.bar(names, macro_f1, color='#1f4e79', width=0.55, zorder=3)
    bars[macro_f1.index(max(macro_f1))].set_color('#375623')

    for bar, f in zip(bars, macro_f1):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.003,
                f'{f:.3f}', ha='center', va='bottom', fontsize=10)

    plt.xticks(rotation=25, ha='right')
    ax.set_title(f'{model.upper()} — Macro F1 Score per Case', pad=10)
    ax.set_ylabel('Macro F1 Score')
    ax.set_ylim(max(0, min(macro_f1) - 0.05), min(1.0, max(macro_f1) + 0.05))

    plt.tight_layout()
    plt.savefig(os.path.join(model, f'{model}_macro_f1_metrics.png'),
                dpi=300, facecolor='white', bbox_inches='tight')
    plt.show()


# ──────────────────────────────────────────────────────────────────────────────
# BEST CASES  — 2 plots total across all models
# ──────────────────────────────────────────────────────────────────────────────
def _plot_best_cases(svm_results=None, rvm_results=None, mlp_results=None):
    """
    Generate exactly 2 plots comparing only the best case from each model:
      Plot 1 — Grouped bar chart: Test Accuracy and Macro F1
      Plot 2 — Confusion matrices side by side (one per model)
    """
    import numpy as np
    import seaborn as sns
    import matplotlib.colors as mc
    from sklearn.metrics import confusion_matrix as sk_cm

    class_names = ['T-Shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot']

    # Collect best result per model (highest test_acc)
    sources = [('SVM', svm_results), ('RVM', rvm_results), ('MLP', mlp_results)]
    bests = {}
    for label, log in sources:
        if log:
            valid = [r for r in log if r is not None and 'test_acc' in r]
            if valid:
                bests[label] = max(valid, key=lambda r: r['test_acc'])

    if not bests:
        print('No results available for best-case plots.')
        return

    models   = list(bests.keys())
    best_res = list(bests.values())
    n        = len(models)

    os.makedirs('best', exist_ok=True)

    # ── Plot 1: Grouped bar — Accuracy + Macro F1 ───────────────────────────
    accs  = [r['test_acc'] for r in best_res]
    f1s   = [r['macro_f1'] * 100 for r in best_res]  # scale to % for same axis
    names = [f"{m}\n({r['name']})" for m, r in zip(models, best_res)]

    x   = np.arange(n)
    w   = 0.35

    fig, ax = plt.subplots(figsize=(max(8, n * 3), 6), facecolor='white')
    _style_ax(ax)

    bars1 = ax.bar(x - w / 2, accs, width=w, label='Test Accuracy (%)',
                   color='#1f4e79', zorder=3)
    bars2 = ax.bar(x + w / 2, f1s,  width=w, label='Macro F1 × 100',
                   color='#833c00', zorder=3)

    for bar, v in zip(bars1, accs):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.4,
                f'{v:.2f}%', ha='center', va='bottom', fontsize=11)
    for bar, v in zip(bars2, f1s):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.4,
                f'{v/100:.4f}', ha='center', va='bottom', fontsize=11)

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=12)
    ax.set_ylabel('Score')
    ax.set_title('Best Case Comparison — Test Accuracy and Macro F1', pad=12)
    ax.set_ylim(max(0, min(accs + f1s) - 5), min(105, max(accs + f1s) + 6))
    ax.legend(facecolor='white', edgecolor='#333333', framealpha=0.9)

    plt.tight_layout()
    plt.savefig('best/best_cases_comparison.png',
                dpi=300, facecolor='white', bbox_inches='tight')
    plt.show()

    # ── Plot 2: Confusion matrices ───────────────────────────────────────────
    cmap = mc.LinearSegmentedColormap.from_list(
        'formal', ['white', '#1a2a6c', '#3b6fd4'], N=256)

    fig, axes = plt.subplots(1, n, figsize=(8 * n, 8), facecolor='white')
    if n == 1:
        axes = [axes]

    for ax, (model_name, r) in zip(axes, bests.items()):
        cm   = sk_cm(r['y_true'], r['y_pred'])
        cm_n = cm.astype(float) / cm.sum(axis=1, keepdims=True)

        sns.heatmap(cm_n, annot=True, fmt='.2f', cmap=cmap,
                    xticklabels=class_names, yticklabels=class_names,
                    vmin=0, vmax=1, linewidth=0.4, linecolor='#dddddd',
                    ax=ax, cbar=True, annot_kws={'size': 9})

        ax.set_title(f"{model_name} — Best Case: {r['name']}  ({r['test_acc']}%)",
                     fontsize=13, pad=10, color='black')
        ax.set_xlabel('Predicted Label', fontsize=12, color='black')
        ax.set_ylabel('True Label', fontsize=12, color='black')
        ax.tick_params(colors='black', labelsize=10)
        ax.set_facecolor('white')

        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=9, colors='black')
        cbar.set_label('Normalised Count', fontsize=10)

    fig.suptitle('Best Case Confusion Matrices', fontsize=15,
                 color='black', y=1.01)
    fig.patch.set_facecolor('white')
    plt.tight_layout()
    plt.savefig('best/best_cases_confusion_matrices.png',
                dpi=300, facecolor='white', bbox_inches='tight')
    plt.show()

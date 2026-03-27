import os
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mc
import matplotlib as mpl

mpl.rcParams['font.family'] = 'Times New Roman'

from sklearn.metrics import confusion_matrix


def _confusion_matrix(model, results):
    class_names = ['T-Shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot']

    cmap = mc.LinearSegmentedColormap.from_list(
        'formal', ['white', '#1a2a6c', '#3b6fd4'], N=256)

    n    = len(results)
    rows = (n + 1) // 2          # 2 columns, ceil(n/2) rows

    fig, axes = plt.subplots(nrows=rows, ncols=2,
                             figsize=(18, 7.5 * rows),   # taller per row
                             facecolor='white')
    axes = axes.flatten()

    for idx, (ax, r) in enumerate(zip(axes, results)):
        cm   = confusion_matrix(r['y_true'], r['y_pred'])
        cm_n = cm.astype(float) / cm.sum(axis=1, keepdims=True)

        sns.heatmap(cm_n, annot=True, fmt='.2f', cmap=cmap,
                    xticklabels=class_names, yticklabels=class_names,
                    vmin=0, vmax=1, linewidth=0.4, linecolor='#dddddd',
                    ax=ax, cbar=True, annot_kws={'size': 9})

        ax.set_title(f"{r['name']}  —  {r['test_acc']}%",
                     color='black', fontsize=13, pad=10)
        ax.set_xlabel('Predicted Label', color='black', fontsize=12)
        ax.set_ylabel('True Label', color='black', fontsize=12)
        ax.tick_params(colors='black', labelsize=10)
        ax.set_facecolor('white')
        fig.patch.set_facecolor('white')

        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=9, colors='black')
        cbar.set_label('Normalised Count', fontsize=10)

    # hide any unused axes (odd number of results)
    for ax in axes[n:]:
        ax.set_visible(False)

    fig.suptitle(f'{model.upper()} — Confusion Matrices',
                 color='black', fontsize=15, y=1.01)
    plt.tight_layout()
    os.makedirs(model, exist_ok=True)
    plt.savefig(os.path.join(model, f'{model}_confusion_matrix.png'),
                dpi=300, facecolor='white', bbox_inches='tight')
    plt.show()

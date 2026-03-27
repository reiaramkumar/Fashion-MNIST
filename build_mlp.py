# .... IMPORT MODULES ....

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from torchvision import datasets, transforms
from torchvision.datasets import FashionMNIST
import time, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

from data import train_loader, val_loader, test_loader



# .... OPTIMIZER SELECTOR ....
def _select_optimizer(name, params, lr, weight_decay):
    _optimizers = {
        'adam':     lambda: optim.Adam(params, lr=lr, weight_decay=weight_decay),
        'adamw':   lambda: optim.AdamW(params, lr=lr, weight_decay=weight_decay),
        'sgd':      lambda: optim.SGD(params, lr=lr, weight_decay=weight_decay),
        'rmsprop':  lambda: optim.RMSprop(params, lr=lr, weight_decay=weight_decay),
        }

    assert name in _optimizers, f"Unknown optimizer: {name}"
    return _optimizers[name]()


# .... EVALUATOR ...
def _evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    all_pred, all_labels = [],[]
    with torch.no_grad():
        for imgs, labels in loader:
            imgs,labels = imgs.to(device), labels.to(device)
            preds = torch.argmax(model(imgs), 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_pred.append(preds.cpu())
            all_labels.append(labels.cpu())
        accuracy = correct / total
        y_pred = torch.cat(all_pred).numpy()
        y_true = torch.cat(all_labels).numpy()
    return accuracy, y_pred, y_true


os.makedirs('results', exist_ok=True)
_results_log = []

# .... MLP BUILDER ....

def _build_mlp(
    input_size = 784,       # 28 * 28 = 784 constant val
    hidden_sizes = [128],   # 'relu' | 'gelu' | 'tanh' | 'sigmoid'| 'leakyrelu'
    activation = 'relu',    #
    dropout = 0.0,
    batch_norm = False,
    output_size = 10,
    ):

    # defining the various activations :)
    # --> we will be using relu n gelu for our 8 cases lmk if u want me to add one of these asw

    _activations = {
        'relu':       nn.ReLU,
        'gelu':       nn.GELU,
        'sigmoid':    nn.Sigmoid,
        'tanh':       nn.Tanh,
        'leaky_relu': nn.LeakyReLU,
    }

    assert activation in _activations, f"Unknown activation function: {activation}"

    layers = [nn.Flatten()]
    prev = input_size

    for width in hidden_sizes:
        layers.append(nn.Linear(prev, width))

        if batch_norm:
            layers.append(nn.BatchNorm1d(width))
        layers.append(_activations[activation]())

        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        prev = width


    layers.append(nn.Linear(prev, output_size))

    return nn.Sequential(*layers)

def _run_mlp(
        name = 'mlp_',
        hidden_size = [128],
        activation = 'relu',
        dropout = 0.0,
        batch_norm = False,
        epochs = 10,
        lr = 0.001,
        optimizer = 'adam',
        weight_decay = 0.0,
        verbose = True,
        plot_curves = True,
):

    print(f"\n{'.'*50}")
    print(f"  {name}")
    print(f"{'.'*50}")


    # my lap doesn't have a gpu but urs might
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # setting gpu/cpu based on sys specs
    model = _build_mlp (hidden_sizes = hidden_size, activation = activation,
                       dropout = dropout, batch_norm = batch_norm).to(device)


    n_params = sum(p.numel() for p in model.parameters())
    opt = _select_optimizer(optimizer, model.parameters(), lr, weight_decay)
    criterion = nn.CrossEntropyLoss()

    scheduler = optim.lr_scheduler.StepLR(opt, step_size = 5, gamma = 0.5)
    history = {'train_loss': [], 'val_acc': [], 'time': []}
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for imgs, labels in train_loader:
            imgs,labels = imgs.to(device), labels.to(device)
            opt.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            opt.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        val_acc, _, _ = _evaluate(model, val_loader, device)
        history['train_loss'].append(avg_loss)
        history['val_acc'].append(val_acc)
        scheduler.step()

        if verbose:
            print(f"  epoch {epoch:>2}/{epochs}  loss={avg_loss:.4f}  val_acc={val_acc*100:.2f}%")

    train_time = time.time() - t0
    test_acc , y_pred,y_true = _evaluate(model, test_loader, device)
    report   = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    macro_f1 = report['macro avg']['f1-score']

    results = dict(
                    name = name,
                    hidden_size = str(hidden_size),
                    activation = activation,
                    dropout = dropout,
                    batch_norm = batch_norm,
                    optimizer = optimizer,
                    lr = lr,
                    epochs = epochs,
                    params = n_params,
                    test_acc = round(test_acc * 100, 2),
                    val_acc = round(val_acc * 100, 2),
                    macro_f1 = round(macro_f1, 4),
                    train_time = round(train_time, 1),
                    y_pred = y_pred,
                    y_true = y_true,
                    history = history,
                    model = model)
    _results_log.append(results)

    print(f"\n{'─'*50}")
    print(f"  {name}")
    print(f"  architecture : {hidden_size}  act={activation}  dropout={dropout}  bn={batch_norm}")
    print(f"  optimizer    : {optimizer}  lr={lr}  wd={weight_decay}")
    print(f"  parameters   : {n_params:,}")
    print(f"  test accuracy: {test_acc*100:.2f}%   macro F1: {macro_f1:.4f}")
    print(f"  training time: {train_time:.1f}s")
    print(f"{'─'*50}\n")
    save_mlp('results')
    return results


def save_mlp(save_dir = None):
    rows = [{k: v for k,v in r.items() if k not in ('y_pred', 'y_true', 'history', 'model') }for r in _results_log ]
    df = pd.DataFrame(rows)
    print(df[['name', 'hidden_size', 'activation', 'dropout', 'batch_norm',
              'optimizer', 'lr', 'params', 'test_acc', 'macro_f1', 'train_time']].to_string(index=False))
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        df.to_csv(os.path.join(save_dir, 'mlp_experiments.csv'), index=False)
        print(f"[saved] {os.path.join(save_dir, 'mlp_experiments.csv')}")
    return df






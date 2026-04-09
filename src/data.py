from torchvision.datasets import FashionMNIST
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
import torch

_tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.2860,), (0.3530,))
])

_full_train = FashionMNIST('./data', train=True, transform=_tf, download=True)
test_dataset = FashionMNIST('./data', train=False, transform=_tf, download=True)

_n_val = 6000
train_dataset, val_dataset = random_split(_full_train, [len(_full_train) - _n_val, _n_val],
                                          generator=torch.Generator().manual_seed(42))
train_loader = DataLoader(train_dataset,batch_size = 64,  shuffle = True,  num_workers = 0)
val_loader   = DataLoader(val_dataset,  batch_size = 256, shuffle = False, num_workers = 0)
test_loader  = DataLoader(test_dataset, batch_size = 256, shuffle = False, num_workers = 0)


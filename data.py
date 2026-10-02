import torch
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor

def build_datasets(root="data", seed=42):
    full_train = FashionMNIST(#train dataset
    root=root,
    train=True,
    download=True,
    transform=ToTensor(),
)
    test_data = FashionMNIST(#test dataset
    root=root,
    train=False,
    download=True,
    transform=ToTensor(),
)
    val_ratio = 1/6
    val_size = int(len(full_train)*val_ratio)
    train_size = len(full_train) - val_size
    train_data, val_data = random_split(
    full_train,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(seed),
)
    return train_data, val_data, test_data


def build_loaders(train_data, val_data, test_data, batch_size=64):
    train_loader = DataLoader(
    dataset = train_data,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,
)
    val_loader = DataLoader(
    dataset = val_data,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0,
)
    test_loader = DataLoader(
    dataset = test_data,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0,
)
    return train_loader, val_loader, test_loader
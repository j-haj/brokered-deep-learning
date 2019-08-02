from enum import Enum
import torch
from torchvision import datasets, transforms

class Dataset(Enum):
    FASHION_MNIST = "fashion_mnist"
    MNIST = "mnist"
    CIFAR10 = "cifar10"

def mnist_loaders(batch_size, test_batch_size=64, train_weight=.9, **kwargs):
    train_set = datasets.MNIST("../data", train=True, download=True,
                              transform=transforms.Compose([
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.1307,), (0.3081,))
                              ]))
    train_n = int(len(train_set) * train_weight)
    val_n = len(train_set) - train_n
    train, val = torch.utils.data.random_split(train_set, [train_n, val_n])
    train_loader = torch.utils.data.DataLoader(train,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               **kwargs)
    val_loader = torch.utils.data.DataLoader(val,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             **kwargs)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST("../data", train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
            ])),
        batch_size=test_batch_size, shuffle=True, **kwargs)

    return train_loader, val_loader, test_loader

def fashion_mnist_loaders(batch_size, test_batch_size=64, train_weight=.9, **kwargs):
    train_set = datasets.FashionMNIST("../data", train=True, download=True,
                                      transform=transforms.Compose([
                                          transforms.ToTensor(),
                                      ]))
    train_n = int(len(train_set) * train_weight)
    val_n = len(train_set) - train_n
    train, val = torch.utils.data.random_split(train_set, [train_n, val_n])
    train_loader = torch.utils.data.DataLoader(train,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               **kwargs)
    val_loader = torch.utils.data.DataLoader(val,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             **kwargs)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST("../data", train=False, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                       ])),
        batch_size=test_batch_size, shuffle=True, **kwargs)

    return train_loader, val_loader, test_loader   

def cifar10_loaders(batch_size, test_batch_size=64, train_weight=.9, **kwargs):
    train_set = datasets.CIFAR10("../data", train=True, download=True,
                                 transform=transforms.Compose([
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5),
                                                          (0.5, 0.5, 0.5))
                                 ]))
    train_n = int(len(train_set)*train_weight)
    val_n = len(train_set) - train_n
    train, val = torch.utils.data.random_split(train_set, [train_n, val_n])
    train_loader = torch.utils.data.DataLoader(train,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               **kwargs)
    val_loader = torch.utils.data.DataLoader(val,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10("../data", train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])),
        batch_size=test_batch_size, shuffle=True, **kwargs)

    return train_loader, val_loader, test_loader

from enum import Enum
import torch
from torchvision import datasets, transforms
from torch.utils.data.sampler import SequentialSampler, SubsetRandomSampler

class Dataset(Enum):
    FASHION_MNIST = "fashion_mnist"
    MNIST = "mnist"
    EMNIST = "emnist"
    KMNIST = "kmnist"
    CIFAR10 = "cifar10"
    CARS = "cars"
    STL10 = "stl10"
    SVHN = "svhn"

def cars_loaders(batch_size, test_batch_size=64, train_weight=.9, **kwargs):
    pass

def svhn_loaders(batch_size, test_batch_size=64, train_weight=.9, **kwargs):
    train_set = datasets.SVHN("../data", split="train", download=True,
                              transform=transforms.Compose([
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.5, 0.5, 0.5),
                                                       (0.5, 0.5, 0.5))
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
        datasets.SVHN("../data", split="test", transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])),
        batch_size=test_batch_size, shuffle=True, **kwargs)
    return train_loader, val_loader, test_loader

def stl10_loaders(batch_size, test_batch_size=64, train_weight=.9, **kwargs):
    train_set = datasets.STL10("../data", download=True,
                               transform=transforms.Compose([
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5),
                                                        (0.5, 0.5, 0.5))
                               ]))
    train_n = int(len(train_set) * train_weight)
    indices = list(range(len(train_set)))
    val_n = len(train_set) - train_n
    #train, val = train_set[:train_n], train_set[train_n:]
    train_indices, val_indices = indices[:train_n], indices[train_n:]
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SequentialSampler(val_indices)
    #train, val = torch.utils.data.random_split(train_set, [train_n, val_n])
    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=batch_size,
                                               sampler=train_sampler,
                                               **kwargs)
    val_loader = torch.utils.data.DataLoader(train_set,
                                             batch_size=batch_size,
                                             sampler=val_sampler,
                                             **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.STL10("../data", split="test", download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                       ])))
    return train_loader, val_loader, test_loader
    
def mnist_loaders(batch_size, test_batch_size=64, train_weight=.9, sub_sample=1.0, **kwargs):
    train_set = datasets.MNIST("../data", train=True, download=True,
                              transform=transforms.Compose([
                                  transforms.ToTensor(),

                              ]))
    train_n = int(len(train_set) * train_weight)
    val_n = len(train_set) - train_n
    train_n = int(sub_sample*train_n)
    misc = len(train_set) - train_n - val_n
    train, val, _ = torch.utils.data.random_split(train_set, [train_n, val_n, misc])
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

            ])),
        batch_size=test_batch_size, shuffle=True, **kwargs)

    return train_loader, val_loader, test_loader

def fashion_mnist_loaders(batch_size, test_batch_size=64, train_weight=.9, sub_sample=1.0, **kwargs):
    train_set = datasets.FashionMNIST("../data", train=True, download=True,
                                      transform=transforms.Compose([
                                          transforms.ToTensor(),
                                      ]))
    train_n = int(len(train_set) * train_weight)
    val_n = len(train_set) - train_n
    train_n = int(sub_sample*train_n)
    misc = len(train_set) - train_n - val_n
    train, val, _ = torch.utils.data.random_split(train_set, [train_n, val_n, misc])
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

def kmnist_loaders(batch_size, test_batch_size=64, train_weight=.9, sub_sample=1.0, **kwargs):
    train_set = datasets.KMNIST("../data", train=True, download=True,
                                transform=transforms.Compose([
                                    transforms.ToTensor(),
                                ]))
    train_n = int(len(train_set) * train_weight)
    val_n = len(train_set) - train_n
    train_n = int(sub_sample*train_n)
    misc = len(train_set) - train_n - val_n
    train, val, _ = torch.utils.data.random_split(train_set, [train_n, val_n, misc])
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

def emnist_loaders(batch_size, test_batch_size=64, train_weight=.9, sub_sample=1.0, **kwargs):
    train_set = datasets.EMNIST("../data", train=True, split="balanced", download=True,
                                transform=transforms.Compose([
                                    transforms.ToTensor(),
                                ]))
    train_n = int(len(train_set) * train_weight)
    val_n = len(train_set) - train_n
    train_n = int(sub_sample*train_n)
    misc = len(train_set) - train_n - val_n
    train, val, _ = torch.utils.data.random_split(train_set, [train_n, val_n, misc])
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

def cifar10_loaders(batch_size, test_batch_size=64, train_weight=.9, sub_sample=1.0, **kwargs):
    train_set = datasets.CIFAR10("../data", train=True, download=True,
                                 transform=transforms.Compose([
                                     transforms.ToTensor(),
                                     transforms.Normalize((125.3/255,123/255,113.5/255),
                                                          (63/255,62.1/255,66.7/255)),
                                 ]))
    train_n = int(len(train_set)*train_weight)
    val_n = len(train_set) - train_n
    train_n = int(sub_sample*train_n)
    misc = len(train_set) - train_n - val_n
    train, val, _ = torch.utils.data.random_split(train_set, [train_n, val_n, misc])
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

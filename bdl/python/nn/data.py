import torch
from torchvision import datasets, transforms

def mnist_loaders(batch_size, test_batch_size=64, **kwargs):
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST("../data", train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch_size, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST("../data", train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
            ])),
        batch_size=test_batch_size, shuffle=True, **kwargs)

    return train_loader, test_loader

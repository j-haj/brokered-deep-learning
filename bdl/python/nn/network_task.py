import logging

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from nn.data import mnist_loaders, fashion_mnist_loaders, cifar10_loaders, Dataset
from nn.classification import SimpleNN
from nn.layer import LayerType, layers_from_string
from result.result import NetworkResult


_TENSOR_SHAPE = {Dataset.MNIST: (1, 28, 28),
                 Dataset.FASHION_MNIST: (1, 28, 28),
                 Dataset.CIFAR10: (3, 32, 32)}

class NetworkTask(object):
    def __init__(self, model, dataset, batch_size, n_epochs=5, log_interval=100, n_modules=3):
        self.model = None
        self.n_modules = n_modules
        self.tensor_shape = _TENSOR_SHAPE[dataset]
        self.layers = model
        self.dataset = dataset
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.log_interval = log_interval
        have_cuda = torch.cuda.is_available()
        if have_cuda:
            logging.debug("CUDA available. Setting device to GPU.")
        self.device = torch.device("cuda" if have_cuda else "cpu")
        self.kwargs = {"num_workers": 1, "pin_memory": True} if have_cuda else {}

    def build_model(self):
        self.model = SimpleNN(self.tensor_shape, 10, self.layers, self.n_modules)
        
    def run(self, cuda_device_id=None):
        if cuda_device_id is not None and torch.cuda.is_available():
            self.device = torch.device("cuda:%d" % cuda_device_id)
        logging.debug("Training network: {}".format(self.layers))
        train_loader, val_loader = self.get_data()
        self.build_model()
        self.model.to(self.device)
        optimizer = optim.Adam(self.model.parameters())
        for epoch in range(self.n_epochs):
            self.train(train_loader, optimizer, epoch)
        val_acc = self.eval(val_loader)
        logging.debug("Finished training %d epochs with %.6f validation accuracy" %
                      (self.n_epochs, val_acc))

        # Clean up resources
        del self.model
        del train_loader
        del val_loader
        
        return NetworkResult(val_acc)

    def get_data(self):
        if self.dataset == Dataset.MNIST:
            train_loader, val_loader, _ = mnist_loaders(self.batch_size,
                                                        **self.kwargs)
        elif self.dataset == Dataset.FASHION_MNIST:
            train_loader, val_loader, _ = fashion_mnist_loaders(self.batch_size,
                                                                **self.kwargs)
        elif self.dataset == Dataset.CIFAR10:
            train_loader, val_loader, _ = cifar10_loaders(self.batch_size,
                                                          **self.kwargs)
        else:
            raise ValueError("unknown dataset: {}".format(self.dataset.value))
        return train_loader, val_loader

    def train(self, train_loader, optimizer, epoch):
        self.model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            optimizer.zero_grad()
            output = self.model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()

            # Log training progress
            if batch_idx % self.log_interval == 0:
                logging.debug("Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))

    def eval(self, val_loader):
        """Evaluates the model on the validation set.

        Args:
        val_loader: validation dataloader

        Return:
        Returns the test accuracy.
        """
        correct = 0.0
        total = 0.0
        with torch.no_grad():
            for (x, y) in val_loader:
                x, y = x.to(self.device), y.to(self.device)
                outputs = self.model(x)
                _, predicted = torch.max(outputs.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
        return correct / total


import logging

import numpy as np
import os

import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision.utils import save_image

from nn.data import mnist_loaders, fashion_mnist_loaders, cifar10_loaders, Dataset
from nn.classification import SimpleNN
from nn.autoencoder import SequentialAE
from nn.layer import LayerType, layers_from_string
from result.result import NetworkResult


_TENSOR_SHAPE = {Dataset.MNIST: (1, 28, 28),
                 Dataset.FASHION_MNIST: (1, 28, 28),
                 Dataset.CIFAR10: (3, 32, 32)}
def get_data(dataset, batch_size, **kwargs):
    if dataset == Dataset.MNIST:
        train_loader, val_loader, _ = mnist_loaders(batch_size,
                                                    **kwargs)
    elif dataset == Dataset.FASHION_MNIST:
        train_loader, val_loader, _ = fashion_mnist_loaders(batch_size,
                                                            **kwargs)
    elif dataset == Dataset.CIFAR10:
        train_loader, val_loader, _ = cifar10_loaders(batch_size,
                                                     **kwargs)
    else:
        raise ValueError("unknown dataset: {}".format(self.dataset.value))
    return train_loader, val_loader


class AENetworkTask(object):

    _model = None
    _device = torch.device("cpu")
    _cuda = False
    _batch_size = 32

    def __init__(self, img_path, layers, dataset, n_epochs=10,
                 log_interval=10, n_modules=2, n_reductions=1,
                 binarize=False, fuzz=False):
        if dataset == Dataset.CIFAR10:
            d = "cifar10"
        elif dataset == Dataset.MNIST:
            d = "mnist"
        elif dataset == Dataset.FASHION_MNIST:
            d = "fashion_mnist"
        else:
            d = "unknown"
        self._img_path = "%s_%s_%dmod_%dred" % (d, img_path, n_modules, n_reductions)
        self._layers = layers
        self._dataset = dataset
        self._n_epochs = n_epochs
        self._log_int = log_interval
        self._n_modules = n_modules
        self._n_reductions = n_reductions
        self._tensor_shape = _TENSOR_SHAPE[dataset]
        self._check_gpu()
        self._kwargs = {"num_workers": 1, "pin_memory": True} if self._cuda else {}
        self._log_interval = 10
        self._binarize = binarize
        self._fuzz = fuzz

        # Make image directory
        p = "./img/{}".format(self._img_path)
        if not os.path.exists(p):
            os.makedirs(p)

    def _check_gpu(self):
        if torch.cuda.is_available():
            self._cuda = True
            logging.debug("CUDA available. Setting device to GPU.")
            self._device = torch.device("cuda")

    def build_model(self):
        self.model = SequentialAE(self._layers, self._tensor_shape,
                                  n_modules=self._n_modules,
                                  n_reductions=self._n_reductions,
                                  binarize=self._binarize,
                                  fuzz=self._fuzz)
        logging.debug("Model {} as {:.1f}M parameters".format(
            self._layers,
            sum(p.numel() for p in self.model.parameters() if p.requires_grad)/1e6))

    def to_img(self, x):
        x = 0.5 * (x + 1)
        x = x.clamp(0, 1)
        x = x.view(x.size(0), self._tensor_shape[0], self._tensor_shape[1],
                   self._tensor_shape[2])
        return x

    def train(self, train_loader, optimizer, epoch):
        self.model.to(self._device)
        self.model.train()
        
        for batch_idx, (img, _) in enumerate(train_loader):

            img = img.to(self._device)
            noise = torch.zeros_like(img)
            if self._fuzz:
                noise = torch.randn_like(img) / 3
            noised_img = img + noise

            output = self.model(noised_img)
            loss = F.mse_loss(output, img)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Log training progress
            if epoch % self._log_interval == 0:
                logging.debug("Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch, batch_idx * len(img), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss))

        if epoch % 1 == 0:
            pic_in = self.to_img(noised_img.cpu().data)
            pic_out = self.to_img(output.cpu().data)
            t = ""

            if self._binarize:
                t = "_b"
            elif self._fuzz:
                t = "_f"
                
            dc_img_path = "./img/{}/dc_img{}_{}.png".format(self._img_path,
                                                            t,
                                                            epoch)
            en_img_path = "./img/{}/en_img{}_{}.png".format(self._img_path,
                                                            t,
                                                            epoch)
            save_image(pic_in, en_img_path)
            save_image(pic_out, dc_img_path)
    
    def run(self, cuda_device_id=None):
        if cuda_device_id is not None and self._cuda:
            self._device = torch.device("cuda:%d" % cuda_device_id)
        logging.debug("Training network: {}".format(self._layers))
        # Load data
        train_loader, val_loader = get_data(self._dataset, self._batch_size, **self._kwargs)

        # Build model
        self.build_model()

        # Move model to GPU or CPU
        self.model.to(self._device)
        
        optimizer = optim.Adam(self.model.parameters())

        # Train
        for epoch in range(self._n_epochs):
            self.train(train_loader, optimizer, epoch)

        # Get validation accuracy
        val_acc = self.eval(val_loader)
        logging.debug("Finished training %d epochs with %.6f validation accuracy" %
                      (self._n_epochs, val_acc))

        # Clean up resources
        del self.model
        del train_loader
        del val_loader

        return NetworkResult(val_acc)

    def eval(self, val_loader):
        """Evaluates the model on the validation set.

        Args:
        val_loader: validation dataloader

        Return:
        Returns the test accuracy.
        """
        loss = 0.0
        with torch.no_grad():
            for (img, _) in val_loader:
                img = img.to(self._device)
                output = self.model(img)
                loss += F.mse_loss(output, img)

        return 1.0/(loss + 1e-9)
    
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
        logging.debug("Model {} as {:.1f}M parameters".format(
            self.layers,
            sum(p.numel() for p in self.model.parameters() if p.requires_grad)/1e6))

    def run(self, cuda_device_id=None):
        if cuda_device_id is not None and torch.cuda.is_available():
            self.device = torch.device("cuda:%d" % cuda_device_id)
        logging.debug("Training network: {}".format(self.layers))
        train_loader, val_loader = get_data(self.dataset, self.batch_size, **self.kwargs)
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

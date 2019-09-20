import logging

import numpy as np
import os

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
from torchvision.utils import save_image

import scipy.stats as stats
from scipy.stats import norm

from nn.data import mnist_loaders, fashion_mnist_loaders, cifar10_loaders, stl10_loaders, Dataset
from nn.classification import SimpleNN
from nn.autoencoder import SequentialAE
from nn.vae import SequentialVAE
from nn.layer import LayerType, layers_from_string
from result.result import NetworkResult


_TENSOR_SHAPE = {Dataset.MNIST: (1, 28, 28),
                 Dataset.FASHION_MNIST: (1, 28, 28),
                 Dataset.CIFAR10: (3, 32, 32),
                 Dataset.SVHN: (3, 32, 32),
                 Dataset.STL10: (3, 96, 96)}
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
    elif dataset == Dataset.STL10:
        train_loader, val_loader, _ = stl10_loaders(batch_size,
                                                    **kwargs)
    else:
        raise ValueError("unknown dataset: {}".format(self.dataset.value))
    return train_loader, val_loader


class AENetworkTask(object):

    def __init__(self, img_path, layers, dataset, n_epochs=10,
                 log_interval=10, n_modules=2, n_reductions=1,
                 binarize=False, fuzz=False):
        if dataset == Dataset.CIFAR10:
            d = "cifar10"
        elif dataset == Dataset.MNIST:
            d = "mnist"
        elif dataset == Dataset.FASHION_MNIST:
            d = "fashion_mnist"
        elif dataset == Dataset.SVHN:
            d = "svhn"
        elif dataset == Dataset.STL10:
            d = "stl10"
        else:
            d = "unknown"
        self._model = None
        self._device = torch.device("cpu")
        self._cuda = False
        self._batch_size = 32
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
            if (batch_idx+1) % (self._log_interval*len(img)) == 0:
                logging.debug("Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch, batch_idx * len(img), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss))

        logging.debug("Saving images.")
        if epoch % 1 == 0:
            pic_in = self.to_img(noised_img.cpu().data)
            pic_out = self.to_img(output.cpu().data)
            t = ""

            if self._binarize:
                t = "_b"
            elif self._fuzz:
                t = "_f"
                
            dc_img_path = "./img/{}/dc_img{}_{}.tiff".format(self._img_path,
                                                             t,
                                                             epoch)
            en_img_path = "./img/{}/en_img{}_{}.tiff".format(self._img_path,
                                                             t,
                                                             epoch)
            save_image(pic_in, en_img_path)
            save_image(pic_out, dc_img_path)
        logging.debug("Done saving images.")
    
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
                img = img.to(self._device)
                noise = torch.zeros_like(img)
                if self._fuzz:
                    noise = torch.randn_like(img)/3
                noised_img = img + noise
                output = self.model(noised_img)
                loss += self.loss_function(output, img)
            pic_noise = self.to_img(noised_img[:16].cpu().data)
            pic_in = self.to_img(img[:16].cpu().data)
            pic_out = self.to_img(output[:16].cpu().data)
            t = ""

            if self._binarize:
                t = "_b"
            elif self._fuzz:
                t = "_f"
                
            dc_img_path = "./img/{}/dc_img{}_{}.tiff".format(self._img_path,
                                                            t,
                                                            self._n_epochs)
            en_img_path = "./img/{}/en_img{}_{}.tiff".format(self._img_path,
                                                            t,
                                                            self._n_epochs)

            if self._fuzz:
                fuzz_path = "./img/{}/fuzzed_img{}_{}.tiff".format(self._img_path,
                                                                   t,
                                                                   self._n_epochs)
                save_image(pic_noise, fuzz_path)

            save_image(pic_in, en_img_path)
            save_image(pic_out, dc_img_path)

        loss = loss.to("cpu")
        return 1.0/(loss + 1e-9)

class VAENetworkTask(object):

    def __init__(self, img_path, layers, dataset, batch_size=32, n_epochs=10, log_interval=10):
        if dataset == Dataset.CIFAR10:
            d = "cifar10"
        elif dataset == Dataset.MNIST:
            d = "mnist"
        elif dataset == Dataset.FASHION_MNIST:
            d = "fashion_mnist"
        elif dataset == Dataset.SVHN:
            d = "svhn"
        elif dataset == Dataset.STL10:
            d = "stl10"
        else:
            d = "unknown"
        self.model = None
        self._device = torch.device("cpu")
        self._cuda = False
        self._batch_size = batch_size
        self._img_path = "%s_%s" % (d, img_path)
        self._layers = layers
        try:
            self._latent_dim = int(layers.split("|")[1])
        except:
            logging.error("Failed to parse: {}".format(layers))
        self._dataset = dataset
        self._n_epochs = n_epochs
        self._log_int = log_interval
        self._tensor_shape = _TENSOR_SHAPE[dataset]
        self._check_gpu()
        self._kwargs = {"num_workers": 1, "pin_memory": True} if self._cuda else {}
        self._log_interval = 10

        # Make image directory
        p = "./img/%s" % self._img_path
        s = "./img/%s/samples" % self._img_path
        m = "./img/%s/manifold" % self._img_path
        if not os.path.exists(p):
            os.makedirs(p)
        if not os.path.exists(s):
            os.makedirs(s)
        if not os.path.exists(m):
            os.makedirs(m)
            
            

    def _check_gpu(self):
        if torch.cuda.is_available():
            self._cuda = True
            logging.debug("CUDA available. Setting device to GPU.")
            self._device = torch.device("cuda")

    def build_model(self):
        self.model = SequentialVAE(self._layers, self._tensor_shape)
        logging.debug("Model {} as {:.2f}M parameters".format(
            self._layers,
            sum(p.numel() for p in self.model.parameters() if p.requires_grad)/1e6))


    def to_img(self, x):
        x = 0.5 * (x + 1)
        x = x.clamp(0, 1)
        x = x.view(x.size(0), *self._tensor_shape)
        return x


    def loss_function(self, recon_x, x, mu, logvar):
        shape = self._tensor_shape[1]*self._tensor_shape[2]
        bce = F.binary_cross_entropy(recon_x, x.view(-1, shape), reduction="sum")
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return bce + kld
    
    def train(self, train_loader, optimizer, epoch):
        self.model.to(self._device)
        self.model.train()

        train_loss = 0
        for batch_idx, (img, _) in enumerate(train_loader):

            img = img.to(self._device)

            output, mu, logvar = self.model(img)
            loss = self.loss_function(output, img, mu, logvar)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            train_loss += loss.item()

            # Log training progress
            if (batch_idx+1) % (self._log_interval*len(img)) == 0:
                logging.debug("Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch, batch_idx * len(img), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))
        logging.info("Epoch train loss: {:.6f}".format(train_loss / len(train_loader.dataset)))

        logging.debug("Saving images.")
        if epoch % 1 == 0:
            pic_in = self.to_img(img.cpu().data)
            pic_out = self.to_img(output.cpu().data)
            t = ""
                
            dc_img_path = "./img/{}/dc_img{}_{}.png".format(self._img_path,
                                                            t,
                                                            epoch)
            en_img_path = "./img/{}/en_img{}_{}.png".format(self._img_path,
                                                            t,
                                                            epoch)
            save_image(pic_in, en_img_path)
            save_image(pic_out, dc_img_path)
        logging.debug("Done saving images.")
        return mu, logvar
    
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
        
        optimizer = optim.Adam(self.model.parameters(), lr=1e-3)

        # Train
        for epoch in range(self._n_epochs):
            mu, logvar = self.train(train_loader, optimizer, epoch)
            self.generate_sample(epoch, mu, logvar)
            if self._latent_dim == 2:
                self.generate_manifold(epoch)

        # Get validation accuracy
        val_acc = self.eval(val_loader)
        logging.debug("Finished training %d epochs with %.6f validation accuracy" %
                      (self._n_epochs, val_acc))

        
        # Clean up resources
        del self.model
        del train_loader
        del val_loader

        return NetworkResult(1/val_acc)

    def generate_sample(self, epoch, mu, logvar):
        latent_dim = int(self._layers.split("|")[1])
        sample = torch.randn(64, latent_dim).to(self._device)
        sample = self.model.decode(sample).cpu()
        save_image(sample.view(64, *self._tensor_shape),
                   "./img/{}/samples/sample_{}.png".format(self._img_path, epoch))

    def generate_manifold(self, epoch):
        nx = 20
        ny = 20
        x_vals = np.linspace(.05, .95, nx)
        y_vals = np.linspace(0.5, .95, ny)

        x_dim = self._tensor_shape[1]
        y_dim = self._tensor_shape[2]
        
        samples = np.empty((nx*self._tensor_shape[1], ny*self._tensor_shape[2]))
        for i, xi in enumerate(x_vals):
            for j, yi in enumerate(y_vals):
                z = np.array([[norm.ppf(xi), norm.ppf(yi)]]).astype("float32")
                z = torch.tensor(z, dtype=torch.float32).to(self._device)
                x = self.model.decode(z).cpu().detach().numpy()
                samples[(nx-i-1)*x_dim:(nx-i)*x_dim, j*y_dim:(j+1)*y_dim] = x[0].reshape(x_dim, y_dim)
                        

        save_image(torch.tensor(samples),
                   "./img/{}/manifold/sample_{}.jpeg".format(self._img_path, epoch))
        
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
                output, mu, logvar = self.model(img)
                loss += self.loss_function(output, img, mu, logvar)
            pic_in = self.to_img(img.cpu().data)
            pic_out = self.to_img(output.cpu().data)
                
            dc_img_path = "./img/{}/dc_img_{}.jpeg".format(self._img_path,
                                                            self._n_epochs)
            en_img_path = "./img/{}/en_img_{}.jpeg".format(self._img_path,
                                                            self._n_epochs)

            save_image(pic_in, en_img_path)
            save_image(pic_out, dc_img_path)

        loss = loss.to("cpu")
        return loss / len(val_loader.dataset)
    
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

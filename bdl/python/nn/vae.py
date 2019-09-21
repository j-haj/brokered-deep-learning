from enum import Enum
import logging

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from nn.layer import LayerType, Layer, layers_from_string

_LAYER_TYPES = [LayerType.CONV_1x1, LayerType.CONV_3x3,
                LayerType.CONV_5x5, LayerType.CONV_7x7,
                LayerType.MAX_POOL, LayerType.AVG_POOL]
_FILTER_SIZES = [32, 64, 128]
_LAYER_SIZES = [i*50 for i in range(1,21)]
_LATENT_DIM_SIZES = [i*10 for i in range(1,11)]

def random_layer():
    return Layer(np.random.choice(_LAYER_TYPES),
                 np.random.choice(_FILTER_SIZES))

class SequentialVAEEvo(object):

    def __init__(self, max_len, latent_dim=None):
        self._max_len = max_len
        self._enc_layers = [100]
        self._dec_layers = [100]
        if latent_dim is None:
            self._latent_dim = np.random.choice(_LATENT_DIM_SIZES)
            self._mutate_latent_dim = True
        else:
            self._latent_dim = latent_dim
            self._mutate_latent_dim = False


    def clone(self):
        c = SequentialVAEEvo(self._max_len, self._latent_dim)
        c._enc_layers = [l for l in self._enc_layers]
        c._dec_layers = [l for l in self._dec_layers]
        return c

    def mutate(self):
        if np.random.rand() < .5:
            # Mutate encoder
            if np.random.rand() < .5 and len(self._enc_layers) < self._max_len:
                # Add layer
                self._enc_layers.append(np.random.choice(_LAYER_SIZES))
            else:
                # Modify existing layer
                idx = np.random.randint(0,len(self._enc_layers))
                self._enc_layers[idx] = np.random.choice(_LAYER_SIZES)
        else:
            # Mutate decoder
            if np.random.rand() < .5 and len(self._dec_layers) < self._max_len:
                self._dec_layers.append(np.random.choice(_LAYER_SIZES))
            else:
                idx = np.random.randint(0, len(self._dec_layers))
                self._dec_layers[idx] = np.random.choice(_LAYER_SIZES)
        if self._mutate_latent_dim and np.random.rand() < .5:
            self._latent_dim = np.random.choice(_LATENT_DIM_SIZES)
                
        return self

    def crossover(self, other):
        c = self.clone()
        c._dec_layers = [l for l in other._dec_layers]
        return c

    def mate(self, other):
        o1 = self.clone() if np.random.rand() < .5 else self.clone().mutate()
        o2 = other.clone() if np.random.rand() < .5 else other.clone().mutate()
        offspring = self.crossover(other)
        return [offspring, o1, o2]

    def __len__(self):
        return len(self._layers)

    def to_string(self):
        return (",".join(["%d" % l for l in self._enc_layers])
                + "|"
                + "%d|" % self._latent_dim
                + ",".join(["%d" % l  for l in self._dec_layers]))



    def __repr__(self):
        return "enc:[{}] [{}] dec:[{}]".format(
            ",".join(["Linear(%d)" % l for l in self._enc_layers]),
            self._latent_dim,
            ",".join(["Linear(%d)" % l for l in self._dec_layers]))
        

class SequentialVAE(nn.Module):

    def __init__(self, layers, tensor_shape):
        super(SequentialVAE, self).__init__()
        layers = layers.split("|")
        output_size = int(layers[1])
        self._encoder_layers = self._build_module(
            input_layers=map(lambda x: int(x), layers[0].split(",")),
            input_size=tensor_shape[1]*tensor_shape[2],
            output_size=output_size,
            encoder=True)
        self._decoder_layers = self._build_module(
            input_layers=map(lambda x: int(x), layers[2].split(",")),
            input_size=output_size,
            output_size=tensor_shape[1]*tensor_shape[2])
        self._tensor_shape = tensor_shape

    def _build_module(self, input_layers, input_size, output_size, encoder=False):
        layers = nn.ModuleList()
        prior_dim = input_size
        for l in input_layers:
            layers.append(nn.Linear(prior_dim, l))
            prior_dim = l
            #layers.append(nn.ReLU(True))
        layers.append(nn.Linear(prior_dim, output_size))
        # In the encoder case we add an additional output layer
        if encoder:
            layers.append(nn.Linear(prior_dim, output_size))
            print("Encoder length: {}".format(len(layers)))
        else:
            print("Decoder length: {}".format(len(layers)))
            
        return layers

    def encode(self, x):
        for l in self._encoder_layers[:-2]:
            x = F.relu(l(x))
        return self._encoder_layers[-2](x), self._encoder_layers[-1](x)

    def decode(self, z):
        for l in self._decoder_layers[:-1]:
            z = F.relu(l(z))
        return torch.sigmoid(self._decoder_layers[-1](z))

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        shape = self._tensor_shape[1]*self._tensor_shape[2]
        mu, logvar = self.encode(x.view(-1, shape))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
    
                      
class SequentialVAEEvoBuilder(object):
    def __init__(self, max_len, latent_dim=None):
        self._max_len = max_len
        self._latent_dim = latent_dim

    def build(self):
        return SequentialVAEEvo(self._max_len, self._latent_dim)

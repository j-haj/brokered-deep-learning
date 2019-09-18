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

def random_layer():
    return Layer(np.random.choice(_LAYER_TYPES),
                 np.random.choice(_FILTER_SIZES))

class SequentialVAEEvo(object):

    def __init__(self, max_len):
        self._max_len = max_len
        self._layers = []
        self._add_random_layer()

    @property
    def layers(self):
        return self._layers

    @property.setter
    def layers(self, l):
        self._layers = l

    def _add_random_layer(self):
        assert len(self._layers) <= self._max_len
        self._layers.append(random_layer())

    def clone(self):
        c = SequentialVAEEvo(self._max_len)
        c.layers = [l for l in self._layers]
        return c

    def mutate(self):
        if np.random.rand() < .5 and len(self) < self._max_len:
            self._add_random_layer()
        else:
            idx = np.random.randint(len(self))
            self._layers[idx] = random_layer()
        return self

    def mate(self, other):
        o1 = self.clone() if np.random.rand() < .5 else self.clone().mutate()
        o2 = other.clone() if np.random.rand() < .5 else other.clone().mutate()

        if len(self) > 1 and len(other) > 1:
            offspring = self.crossover(other)
            return [offspring, o1, o2]

        return [o1, o2]

    def crossover(self, other):
        assert len(other) > 1 and len(self) > 1
        s_idx = np.random.randint(len(self))
        o_idx = np.random.randint(len(other))

        c = self.clone()

        c.layers = ([l for l in self._layers[:s_idx]]
                    + [l for l in other.layers[o_idx:]])
        if len(c) > self._max_len:
            c.layers = c.layers[:c._max_len]
        return c

    def __len__(self):
        return len(self._layers)

    def to_string(self):
        return ",".join(["%s:%d" % (l.layer_type.value, l.filter_size)
                         for l in self._layers])

    def __repr__(self):
        return "[{}]".format(",".join([str(l) for l in self._layers]))
        

class SequentialVAE(nn.Module):

    def __init__(self, encoder_layers, decoder_layers, tensor_shape,
                 n_modules=1):
        self._encoder_layers = self._build_module(
            layers_from_string(encoder_layers))
        self._decoder_layers = self._build_module(
            layers_from_string(decoder_layers))
        self._tensor_shape = tensor_shape
        self._n_modules = n_modules
        self._encoder_layers = nn.ModuleList()
        self._decoder_layers = nn.ModuleList()
        self._encoder = None
        self._decoder = None
        self._mu = None
        self._sigma = None

    def _build_module(self, layers, input_size):
        layers = nn.ModuleList()
        prior_chans = input_size
        for l in self._layers:
            layers.append(l.layer(prior_chans))
            prior_chans = l.layer_size
            layers.append(nn.ReLU(True))
        return layers
                          

    def _build_vae(self):
        n_chans = self._tensor_shape[0]
        # Build encoder
        self._encoder = nn.Sequential(*self._encoder_layers)
        
        # Build decoder
        self._decoder = nn.Sequential(*self._decoder_layers)

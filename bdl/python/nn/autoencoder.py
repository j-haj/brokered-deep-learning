from enum import Enum
import logging

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from nn.layer import LayerType, Layer

_LAYER_TYPES = [LayerType.CONV_1x1, LayerType.CONV_3x3,
                LayerType.CONV_5x5, LayerTbype.CONV_7x7]
_FILTER_SIZES = [16, 32, 64, 128]

class EvoAE(nn.Module):

    # A unit is a repeated number of modules followed by a reduction layer.
    _n_units = 3

    # Used to track whether the model has been built. A model must be built
    # before being used.
    _has_been_built = False

    def __init__(self, layers, tensor_shape, max_depth, max_width, n_modules):
        super(EvoAE, self).__init__()
        self._layers = layers
        self._tensor_shape = tensor_shape
        self._max_module_depth = max_depth
        self._max_module_width = max_width
        self._n_modules = n_modules

    def _build_model(self):
        pass

    def __test_forward_and_finalize(x)
        pass

    @property
    def n_units(self):
        return self._n_units
    
    def build(self):
        self._build_model()
        x = torch.rand(self._tensor_shape).unsqueeze(0)
        self.__test_forward_and_finalize(x)
        self._has_been_built = True

    def forward(self, x):
        assert self._has_been_built
        

class WideEvo(object):

    # Maximum number of layers in a given module
    _max_module_depth = 5

    # Max number of channels in a module
    _max_module_width = 5

    _layers = []

    def __init__(self, max_module_depth, max_module_width):
        self._max_module_depth = max_module_depth
        self._max_module_width = max_module_depth


    def _add_random_layer(self, row=-1, col=-1):
        assert (not (row < 0 and col >= 0))

        global _LAYER_TYPES
        global _FILTER_SIZES
        layer_type = np.random.choice(_LAYER_TYPES)
        filter_size = np.random.choice(_FILTER_SIZES)
        layer = Layer(layer_type, filter_size)
        if row < 0:
            self._layers.append([layer])
        elif row >= 0 and col < 0:
            self._layers[row].append(layer)
        else:
            assert row < len(self._layers) and col < len(self._layers[row])
            self._layers[row][col] = layer
        
    def clone(self):
        c = WideEvo(self._max_module_depth, self._max_module_width)
        c._layers = [l for l in self._layers]
        return c

    def mutate(self):
        if len(self._layers) < self.max_module_depth:
            # Either create a new channel or append to an existing one
            if np.random.rand() < .5:
                # Append to existing channel
                r = np.random.randint(self.nrows())
                self._add_random_layer(r)
            else:
                # Append new channel
                self._add_random_layer()
        else:
            r = np.random.randint(self.nrows())
            c = np.random.randint(self.nlayers(r))
            self._add_random_layer(r, c)
        return self

    def nrows(self):
        return len(self._layers)

    def nlayers(self, row):
        assert row < len(self._layers)
        return len(self._layers[row])

    def __len__(self):
        return sum([len(l) for l in self._layers])

    def __repr__(self):
        o = ""
        for ls in self._layers:
            o += "".join(["[%s]" % str(l) for l in ls])
            o += "\n"
        return o
    

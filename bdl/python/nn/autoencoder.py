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
_FILTER_SIZES = [16, 32, 64, 128]

class Autoencoder(nn.Module):

    # A unit is a repeated number of modules followed by a reduction layer.
    _n_units = 3

    # Used to track whether the model has been built. A model must be built
    # before being used.
    _has_been_built = False

    _trainable_layers = []

    _layers = []

    _out_channel = 0

    def __init__(self, layers, tensor_shape, max_depth, max_width, n_modules, n_reductions):
        """Intialize trainable autoencoder with 2D layer module.
        
        """
        super(Autoencoder, self).__init__()
        self._tensor_shape = tensor_shape
        self._max_module_depth = max_depth
        self._max_module_width = max_width
        self._n_modules = n_modules
        self._n_reductions = n_reductions
        self._layer_descriptions = [layers_from_string(l) for l in layers]

    def _build_model(self):
        input_chans = self._tensor_shape[0]
        for i in range(self._n_reductions):
            (m, out_chans) = self._assemble_module(input_chans)
            for j in range(self._n_modules):
                self._layers += m
            # Add reduction layer
            # The input number of chans is the sum of out_chans
            input_chans = sum(out_chans)
            self._layers.append(nn.Conv2d(input_chans, 2*input_chans, 1))
            self._layers.append(F.relu)
            self._layers.append(nn.MaxPool2d(2, stride=2))
            input_chans *= 2

        # Last layer is a 1x1 feature map with a single filter
        self._layers.append(nn.Conv2d(input_chans, 1, 1))
        self._layers.append(F.relu)

    def _assemble_module(self, input_chans):
        module = []
        out_chans = []
        for row in self._layer_descriptions:
            prior_chans = input_chans
            layers = []
            for c in row:
                if c.layer_type == LayerType.CONV_1x1:
                    layers.append(nn.Conv2d(prior_chans,
                                            l.filter_size,
                                            1))
                elif c.layer_type == LayerType.CONV_3x3:
                    layers.append(nn.Conv2d(prior_chans,
                                            l.filter_size,
                                            3,
                                            padding=2))
                elif c.layer_type == LayerType.CONV_5x5:
                    layers.append(nn.Conv2d(prior_chans,
                                            l.filter_size,
                                            5,
                                            padding=4))
                elif c.layer_type == LayerType.MAX_POOL:
                    layers.append(nn.MaxPool2d(2,
                                               stride=1,
                                               padding=1))
                elif c.layer_type == LayerType.AVG_POOL:
                    layers.append(nn.AvgPool2d(2,
                                               stride=1,
                                               padding=1))
                else:
                    raise ValueError("{} - unknown layer type.".format(c))
                layers.append(F.relu)
                prior_chans = l.filter_size
            module.append(layers)
            out_chans.append(prior_chans)
        return (module, out_chans)
                    

            
                    

    def __test_forward_and_finalize(x)
        pass

    def forward(self, x):
        # Note: use `isinstance(v, list)` to check if the current
        # element is a module or part of the reduction layer
        for l in self._layers:
            if isinstance(l, list):
                pass
            else:
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
        

class WideNetBuilder(object):
    def __init__(self, max_depth, max_width):
        self._max_depth = max_depth
        self._max_width = max_width

    def build(self):
        return WideEvo(self._max_depth, self._max_width)
        
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
        if len(self._layers) < self.max_module_width:
            # Either create a new channel or append to an existing one
            r = np.random.randint(self.nrows())
            if (np.random.rand() < .5
                and len(self._layers[r]) < self._max_module_depth):
                # Append to existing channel
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
    

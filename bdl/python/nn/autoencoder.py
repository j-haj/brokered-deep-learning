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

class WideModule(nn.Module):
    """
    A wide module is a single module used to build an autoencoder
    """
    pass

class WideAE(nn.Module):



    def __init__(self, layers, tensor_shape, n_modules):
        """Intialize trainable autoencoder with 2D layer module.

        """
        super(Autoencoder, self).__init__()
        # A unit is a repeated number of modules followed by a reduction layer.
        self._n_units = 2

        # Used to track whether the model has been built. A model must be built
        # before being used.
        self._has_been_built = False
        self._trainable_layers = []
        self._layers = []
        self._out_channel = 0

        # Max width of a module
        self._max_width = 4

        # Max depth of a module (max length of a channel)
        self._max_depth = 4
        self._tensor_shape = tensor_shape
        self._n_modules = n_modules
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





    def __test_forward_and_finalize(x):
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


class SequentialAE(nn.Module):



    def __init__(self, layers, tensor_shape, n_modules,
                 binarize=False, fuzz=False, n_reductions=None):
        super(SequentialAE, self).__init__()
        self._encoder_layers = nn.ModuleList()
        self._decoder_layers = nn.ModuleList()
        self._encoder = None
        self._decoder = None
        self._noise = None
        self._n_reductions = 1
        self._layers = layers_from_string(layers)
        self._tensor_shape = tensor_shape
        self._n_modules = n_modules
        self._should_binarize = binarize
        self._should_fuzz = fuzz
        if n_reductions is not None:
            if n_reductions > 2:
                logging.warn(("Too many reductions can cause "
                              "total loss of information."))
            self._n_reductions = n_reductions

        print("Building autoencoder")
        self._build_autoencoder()
        print("Autencoder built.\n{}\n{}".format(self._encoder, self._decoder))
        
            

    def _build_autoencoder(self):
        # Build encoder
        n_chans = self._tensor_shape[0]
        for i in range(self._n_reductions):
            for _ in range(self._n_modules):
                (layers, n_chans) = self._build_module(n_chans)
                self._encoder_layers.extend(layers)
            self._encoder_layers.extend([
                    nn.Conv2d(n_chans, n_chans*2, 1),
                    nn.ReLU(True),
                    nn.MaxPool2d(2, stride=2)
            ])
            n_chans *= 2

        # Build decoder
        for i in range(self._n_reductions):
            for _ in range(self._n_modules):
                (layers, n_chans) = self._build_module(n_chans)
                self._decoder_layers.extend(layers)
            self._decoder_layers.extend([
                    nn.ConvTranspose2d(n_chans, n_chans, 4, stride=2, padding=1),
                    nn.ReLU(True)
            ])

        self._decoder_layers.extend([
                nn.Conv2d(n_chans, self._tensor_shape[0], 1),
                nn.Tanh()
        ])
        self._encoder = nn.Sequential(*self._encoder_layers)
        self._decoder = nn.Sequential(*self._decoder_layers)

    def forward(self, x):
        x = self._encoder(x)
        if self._should_binarize:
            x = x.round()
        x = self._decoder(x)
        return x

    def _build_module(self, input_size):
        """Builds a module, returning the layers of the modules and the
        output channel size.
        """
        layers = nn.ModuleList()
        prior_chans = input_size
        for l in self._layers:
            if l.layer_type == LayerType.CONV_1x1:
                layers.append(nn.Conv2d(prior_chans,
                                        l.filter_size,
                                        1))
                layers.append(nn.ReLU(True))
                prior_chans = l.filter_size
            elif l.layer_type == LayerType.CONV_3x3:
                layers.append(nn.Conv2d(prior_chans,
                                        l.filter_size,
                                        3,
                                        padding=1))
                layers.append(nn.ReLU(True))
                prior_chans = l.filter_size
            elif l.layer_type == LayerType.CONV_5x5:
                layers.append(nn.Conv2d(prior_chans,
                                        l.filter_size,
                                        5,
                                        padding=2))
                layers.append(nn.ReLU(True))
                prior_chans = l.filter_size
            elif l.layer_type == LayerType.CONV_7x7:
                layers.append(nn.Conv2d(prior_chans,
                                        l.filter_size,
                                        7,
                                        padding=3))
                layers.append(nn.ReLU(True))
                prior_chans = l.filter_size
            elif l.layer_type == LayerType.DROPOUT:
                layers.append(nn.Dropout2d())
            else:
                raise ValueError("{} - unrecognized layer type".format(l))
        return (layers, prior_chans)

class WideNetBuilder(object):
    def __init__(self, max_depth, max_width):
        self._max_depth = max_depth
        self._max_width = max_width

    def build(self):
        return WideEvo(self._max_depth, self._max_width)

class SequentialAEEvoBuilder(object):
    def __init__(self, max_length):
        self._max_length = max_length

    def build(self):
        return SequentialAEEvo(self._max_length)

class SequentialAEEvo(object):


    def __init__(self, max_len=None):
        self._available_layers = [LayerType.CONV_3x3, LayerType.CONV_1x1,
                                  LayerType.CONV_5x5, LayerType.CONV_7x7,
                                  LayerType.DROPOUT]

        self._available_sizes = [8, 16, 32, 64]

        self._layers = []        
        if max_len is not None:
            assert isinstance(max_len, int)
            self._max_length = max_len
        self._add_random_layer()


    def _add_random_layer(self):
        assert len(self._layers) <= self._max_length
        l = Layer(np.random.choice(self._available_layers),
                  np.random.choice(self._available_sizes))
        self._layers.append(l)

    def clone(self):
        c = SequentialAEEvo(self._max_length)
        c._layers = [l for l in self._layers]
        return c

    def mutate(self):
        if len(self._layers) < self._max_length:
            self._add_random_layer()
        else:
            idx = np.random.randint(len(self))
            l = Layer(np.random.choice(self._available_layers),
                      np.random.choice(self._available_sizes))
            self._layers[idx] = l
        return self

    def crossover(self, other):
        assert len(other) > 1 and len(self._layers) > 1
        s_idx = np.random.randint(len(self._layers))
        o_idx = np.random.randint(len(other))
        c = self.clone()
        c._layers = ([l for l in self._layers[:s_idx]]
                     + [l for l in other._layers[o_idx:]])
        if len(c) > self._max_length:
            c._layers = c._layers[:c._max_length]
        return c

    def to_string(self):
        return ",".join(["%s:%d" % (l.layer_type.value, l.filter_size)
                         for l in self._layers])

    def __len__(self):
        return len(self._layers)

    def __repr__(self):
        o = "[{}]".format(",".join([str(l) for l in self._layers]))
        return o



class WideEvo(object):

    # Maximum number of layers in a given module
    _max_module_depth = 4

    # Max number of channels in a module
    _max_module_width = 4

    _layers = []

    def __init__(self, max_module_depth=None, max_module_width=None):
        if max_module_depth is not None:
            self._max_module_depth = max_module_depth
        if max_module_width is not None:
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
        if len(self._layers) < self._max_module_width:
            if len(self._layers) == 0:
                self._add_random_layer()
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

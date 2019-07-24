from enum import Enum
import logging

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision import datasets, transforms
from genotype import Genotype
from layer import LayerType, Layer

_LAYER_TYPES = [LayerType.CONV_3x3, LayerType.CONV_3x1_1x3,
                LayerType.CONV_5x5, LayerType.CONV_5x1_1x5]
_FILTER_SIZES = [10, 50, 100]


class SimpleNN(nn.Module):
    def __init__(self, input_channels, output_size, layers, n_modules):
        super(SimpleNN, self).__init__()
        self.input_size = input_channels
        self.output_size = output_size
        self.n_modules = n_modules
        self.layer_descriptions = layers
        self.layers = []
        self.out_channels = [input_channels]

    def build_model(self):
        for k in range(self.n_modules):
            for (i, l) in enumerate(self.layer_descriptions):
                i += k * len(self.layer_descriptions)
                if l.layer_type == LayerType.CONV_3x3:
                    self.layers.append(nn.Conv2d(self.out_channels[i],
                                                     l.filter_size,
                                                     3))
                elif l.layer_type == LayerType.CONV_3x1_1x3:
                    self.layers.append(nn.Conv2d(self.out_channels[i],
                                                     self.out_channels[i],
                                                     (3,1)))
                    self.layers.append(nn.Conv2d(self.out_channels[i],
                                                     l.filter_size,
                                                     (1,3)))
                elif l.layer_type == LayerType.CONV_5x5:
                    self.layers.append(nn.Conv2d(self.out_channels[i],
                                                     l.filter_size,
                                                     5))
                elif l.layer_type == LayerType.CONV_5x1_1x5:
                    self.layers.append(nn.Conv2d(self.out_channels[i],
                                                     self.out_channels[i],
                                                     (5,1)))
                    self.layers.append(nn.Conv2d(self.out_channels[i],
                                                     l.filter_size,
                                                     (1, 5)))
                else:
                    raise ValueError("{} - unknown layer type".format(l))
                self.out_channels.append(l.filter_size)
            self.out_channels.append(l.filter_size)
            self.layers.append(nn.Conv2d(self.out_channels[-1],
                                             2*self.out_channels[-1],
                                             1))
            o = self.out_channels[-1] * 2
            self.layers.append(nn.MaxPool2d((3,3), stride=2))
            self.out_channels.append(o)
            self.layers.append(F.relu)



    def forward(self, x):
        for l in self.layers:
            x = l(x)
        n_param = np.product(x.size())
        x = x.view(-1, n_param)
        self.layers.append(nn.Linear(n_param, n_param//2))
        x = self.layers[-1](x)
        self.layers.append(nn.Linear(n_param//2, self.output_size))
        return F.log_softmax(x, dim=1)
    
class SimpleEvo(object):
    def __init__(self, max_num_layers):
        self.max_num_layers = max_num_layers
        self.layers = []

    def clone(self):
        c = SimpleEvo(self.max_num_layers)
        c.layers = [l for l in self.layers]
        return c

    def __len__(self):
        return len(self.layers)
        
    def mutate(self):
        global _LAYER_TYPES
        global _FILTER_SIZES
        layer_type = np.random.choice(_LAYER_TYPES)
        layer_size = np.random.choice(_FILTER_SIZES)
        layer = Layer(layer_type, layer_size)
        if (len(self.layers) == 0
            or (len(self.layers) < self.max_num_layers
                and np.random.rand() < .5)):
            self.layers.append(layer)
        else:
            idx = np.random.randint(len(self.layers))
            self.layers[idx] = layer
        return self

    def crossover(self, other):
        assert len(self) == len(other)
        idx = np.random.randint(len(self.layers))
        o = self.clone()
        for i in range(idx, len(self)):
            o.layers[i] = other.layers[i]
        return o
    
    def mate(self, other):
        if len(self) != len(other):
            return [self.clone().mutate(), other.clone().mutate()]

        offspring = self.crossover(other)
        offspring.mutate()
        return [offspring]


    def __repr__(self):
        out = "[IN]{}[OUT]".format(
            "".join([str(l) for l in self.layers]))
        return out
    
class EvoNet(object):

    def __init__(self, max_num_layers, mutation_p, crossover_p):
        self.max_num_layers = max_num_layers
        self.mutation_p = mutation_p
        self.crossover_p = crossover_p
        self.layers = []
        self.connections = np.array([[]])

    def clone(self):
        """Clone creates a deepcopy of self without having to actually call
        deepcopy, which can be slow.

        Returns:
        A copy of self.
        """
        o = EvoNet(self.max_num_layers, self.mutation_p, self.crossover_p)
        o.layers = self.layers[:]
        o.connections = np.zeros(self.connections.shape)
        for i in range(len(layers)):
            for j in range(len(layers)):
                o.connections[i, j] = self.connections[i, j]
        return o

    def mutate(self):
        """Mutate appends a random layer type and adds a random connection
        to a prior layer. This method mutates the caller.
        """
        global _LAYER_TYPES
        layer_type = np.random.choice(_LAYER_TYPES)

        self.layers.append(layer_type)
        row = len(o.layers) - 1
        col = np.random.randint(0, len(o.layers) - 1)
        self.connections[row, col] = 1

    def crossover(self, other):
        assert len(self.layers) == len(other.layers)
        assert self.shape == other.shape
        o = self.clone()
        idx = np.random.randint(0, len(self.layers))
        for i in range(idx, len(o.layers)):
            o.layers[i] = other.layers[i]
            for j in range(idx, len(o.layers)):
                o.connections[i, j] = other.connections[i, j]
        return o
        
    def mate(self, other):
        """Produces either one or two offspring depending on whether
        self and other have the same number of layers. If they do
        have the same number of layers, they perform crossover and
        the offspring is mutated. Otherwise each of self and other
        produce a mutated version of themselves.

        Args:
        other: other EvoNet to mate with.

        Returns:
        An array containing the offspring.
        """
        if len(self.layers) != len(other.layers):
            return [self.clone().mutate(), other.clone().mutate()]

        offspring = self.crossover(other)
        offspring.mutate()
        return [offspring]

    def __repr__(self):
        out = "[IN]{}[OUT]".format(
            "".join(["["+l.value+"]" for l in self.layers]))
        return out



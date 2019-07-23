from enum import Enum
import logging

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision import datasets, transforms
from genotype import Genotype

class LayerType(Enum):
    LAYER_3x3 = "3x3conv"
    LAYER_3x1_1x3 = "3x1conv-1x3conv"
    LAYER_5x5 = "5x5conv"
    LAYER_5x1_1x5 = "5x1conv-1x5conv"
    REDUCTION = "reduction"

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
        o = EvoNet(self.max_num_layers, self.mutation_p, self.crossover_p):
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
        layer_type = np.random.choice([LayerType.LAYER_3x3,
                                       LayerType.LAYER_3x1_1x3,
                                       LayerType.LAYER_5x5,
                                       LayerType.LAYER_5x1_1x5,
                                       LayerType.REDUCTION])

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

        offspring = offspring.mutate()

        return [offspring]

    def __repr__(self):
        return ("[IN]"
                "".join(["["+l.value+"]" for l in self.layers])
                "[OUT]")


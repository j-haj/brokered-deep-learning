from enum import Enum
import logging

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision import datasets, transforms

class LayerType(Enum):
    LAYER_3x3 = "3x3"
    LAYER_3x1_1x3 = "3x1-1x3"
    REDUCTION = "reduction"

class EvoNet(object):

    def __init__(self, max_num_layers, mutation_p, crossover_p):
        self.max_num_layers = max_num_layers
        self.mutation_p = mutation_p
        self.crossover_p = crossover_p
        self.layers = []

    def mutate(self):
        layer_type = np.random.choice([LayerType.LAYER_3x3,
                                       LayerType.LAYER_3x1_1x3,
                                       LayerType.REDUCTION])
        self.layers.append(layer_type)



from enum import Enum


class LayerType(Enum):
    CONV_3x3 = "conv3x3"
    CONV_3x1_1x3 = "conv3x1_1x3"
    CONV_5x5 = "conv5x5"
    CONV_5x1_1x5 = "conv5x1_1x5"
    CONV_1x1 = "conv1x1"

class Layer(object):
    def __init__(self, layer_type, filter_size):
        self.layer_type = layer_type
        self.filter_size = filter_size

    def __repr__(self):
        if self.layer_type == LayerType.REDUCTION:
            return "Layer(Reduction)"
        return "Layer({}x{})".format(self.layer_type.value, self.filter_size)

from enum import Enum


class LayerType(Enum):
    CONV_3x3 = "33"
    CONV_3x1_1x3 = "3113"
    CONV_5x5 = "55"
    CONV_5x1_1x5 = "5115"
    CONV_1x1 = "11"

    @staticmethod
    def from_string(s):
        if s.startswith("33"):
            return LayerType.CONV_3x3
        elif s.startswith("3113"):
            return LayerType.CONV_3x1_1x3
        elif s.startswith("55"):
            return LayerType.CONV_5x5
        elif s.startswith("5115"):
            return LayerType.CONV_5x1_1x5
        elif s.startswith("11"):
            return LayerType.CONV_1x1
        else:
            raise ValueError("Unrecognized layer string: %s" % s)

        
class Layer(object):
    def __init__(self, layer_type, filter_size):
        self.layer_type = layer_type
        self.filter_size = filter_size

    def __repr__(self):
        return "Layer({}x{})".format(self.layer_type.value, self.filter_size)

    @staticmethod
    def from_string(s):
        components = s.split(":")
        layer_type = components[0]
        filter_size = components[1]
        return Layer(LayerType.from_string(layer_type),
                     int(filter_size))

def layers_from_string(s):
    """Takes a string of the form "33:20,3113:10" (etc.) and creates a list
    of layers [conv3x3(20), conv3x1_1x3(10)] etc.

    Args:
    s: string representing layers

    Return:
    A list of Layer objects.
    """
    layers = []
    layer_strs = s.split(",")
    for ls in layer_strs:
        if ls == "":
            continue
        layers.append(Layer.from_string(ls))

    return layers

    

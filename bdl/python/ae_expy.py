import logging
import time

import numpy as np

import torch

from nn.data import Dataset
from nn.autoencoder import SequentialAEEvo, SequentialAE
from nn.network_task import AENetworkTask, VAENetworkTask
from nn.layer import Layer, LayerType

_FILTER_SIZES = [8, 16, 32,  64]
_LAYER_TYPES = [LayerType.CONV_1x1, LayerType.CONV_3x3, LayerType.CONV_5x5,
                LayerType.CONV_7x7, LayerType.DROPOUT]

def custom_layers():
    return [
        Layer(LayerType.CONV_7x7, 16),
        Layer(LayerType.DROPOUT, 8),
        Layer(LayerType.CONV_7x7, 16),
    ]

def random_layers(size):
    layers = []
    for i in range(size):
        layers.append(Layer(np.random.choice(_LAYER_TYPES),
                            np.random.choice(_FILTER_SIZES)))
    return layers

def random_search():
    max_module_len = 10
    epochs = 20
    n_modules = 1
    n_reductions = 3
    n_trials = 30
    fuzz = False
    results = []
    times = []
    for i in range(n_trials):
        s = SequentialAEEvo(max_len=max_module_len)
        n_layers = np.random.randint(1, max_module_len+1)
        s.layers = random_layers(n_layers)
        
    
        task = VAENetworkTask("test", s.to_string(), Dataset.MNIST, n_epochs=epochs)
                
        start = time.time()
        r = task.run()
        times.append(time.time() - start)
        avg_time = np.average(times)
        print("{} of {}. Average time per run: {:.4f}. Estimated {:.4f} remaining.".format(
            i+1, n_trials, avg_time, avg_time * (n_trials - i - 1)))
        results.append(r.accuracy())
    print("Average: {:.4f}\nStd Dev: {:.4f}".format(np.average(results),
                                                    np.std(results, ddof=1)))

def custom_network():
    layers = [
        Layer(LayerType.CONV_7x7, 32),
    ]
    s = "400,400|2|400,400"
    n_epochs = 10
    task = VAENetworkTask("2epoch-c", s, Dataset.MNIST, n_epochs=n_epochs, batch_size=128)
    task.build_model()
    r = task.run()
    print("Result after {} epochs: {}".format(n_epochs, 1/r.accuracy()))

    
def main():
#    s = SequentialAEEvo(max_len=5)
#    for i in range(5):
#        s.mutate()
#        print("iteration %d" % i)
#        print(s)
#
    torch.manual_seed(1)
    debug = True
    fmt_str = "[%(levelname)s][%(filename)s][%(funcName)s][%(lineno)d][%(message)s]"
    if debug:
        logging.basicConfig(format=fmt_str, level=logging.DEBUG)
        logging.debug("Using DEBUG log level.")
    else:
        logging.basicConfig(format=fmt_str, level=logging.INFO)
    #random_search()
    custom_network()

if __name__ == "__main__":
    
    main()

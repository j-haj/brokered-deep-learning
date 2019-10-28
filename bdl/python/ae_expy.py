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

def vae_random_search():
    layer_sizes = [i*50 for i in range(1, 21)]
    latent_dim_sizes = [i*10 for i in range(1, 11)]
    epochs = 20
    n_trials = 40
    results = []
    times = []
    for i in range(n_trials):
        encoder_size = np.random.randint(1,5)
        encoder = np.random.choice(layer_sizes, encoder_size, replace=True)
        decoder_size = np.random.randint(1,5)
        decoder = np.random.choice(layer_sizes, decoder_size, replace=True)
        latent_dim = np.random.choice(latent_dim_sizes)
        s = (",".join([str(x) for x in encoder]) + "|" + str(latent_dim) + "|" +
             ",".join([str(x) for x in decoder]))
        task= VAENetworkTask("random-2poch", s, Dataset.MNIST, n_epochs=epochs)

        start = time.time()
        r = task.run()
        times.append(time.time() - start)
        avg_time = np.average(times)
        print("{} of {}. Average time per run: {:.4f}. Estimated {:.4f} remaining.".format(
            i+1, n_trials, avg_time, avg_time * (n_trials - i - 1)))
        results.append(r.accuracy())
    print("Average: {:.4f}\nStd Dev: {:.4f}".format(np.average(results),
                                                    np.std(results, ddof=1)))

    
def random_search():

    max_module_len = 5
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


class EpochCallback():
    def __init__(self):
        self.losses = []

    def __call__(self, loss):
        self.losses.append(loss.item())
    
def custom_network():

    layers = [
        Layer(LayerType.CONV_7x7, 32),
    ]
    # MNIST
    # Worst 5 epoch
    #s = "850,700,400|30|850,50,1000,900,50"
    # Worst 2 epoch
    #s = "200,500|20|100,650,50"
    # Best 5 epoch
    #s = "800,900|30|150,1000"
    # Best 2 epoch
    #s = "1000|20|1000"

    ## Fashion MNIST
    # Worst 5 epoch
    #s = "900,50,700|60|50,750,600,950"
    # Best 5 epoch
    #s = "1000|2|700,750"

    # Worst 2 epoch
    #s = "950,850,250,200|10|150,500,750"
    # Best 2 epoch
    #s = "1000|2|350,1000"

    s = "900|40|150,750"
    
    n_epochs = 100
    callback = EpochCallback()
    task = VAENetworkTask("best2", s, Dataset.MNIST, n_epochs=n_epochs, batch_size=128,
                          epoch_callback=callback)
    task.build_model()
    r = task.run()
    for (i, l) in enumerate(callback.losses):
        print("{},{:.4f}".format(i+1,l))

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
    #vae_random_search()

if __name__ == "__main__":
    
    main()

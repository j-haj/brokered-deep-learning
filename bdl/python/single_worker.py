import argparse
import logging
import time

import numpy as np
import torch

from model_runner.model_runner import EvoBuilder
from nn.data import mnist_loaders, fashion_mnist_loaders, cifar10_loaders, Dataset
from nn.classification import SimpleNN
from nn.genotype import Population
from nn.layer import LayerType, layers_from_string
from nn.network_task import NetworkTask

_DATASETS = {"fashion_mnist": Dataset.FASHION_MNIST,
             "mnist": Dataset.MNIST,
             "cifar10": Dataset.CIFAR10}

def append_to_file(accuracies, path="single_model_results.csv"):
    with open(path, "a") as f:
        for r in accuracies:
            f.write("{},{:.4f},{:.8f},{}\n".format(r[0], r[1], r[2], r[3]))

def run(population, dataset, n_epochs, result_path,
        cuda_device_id=None, n_generations=100, n_modules=3):
    start = time.time()
    accuracies = []
    initial_size = len(population.population)
    seen = set()
    for generation in range(n_generations):
        logging.info("Generation %d" % (generation+1))
        # Evolve population
        logging.debug("Generating population offspring.")
        population.generate_offspring()
        
        # Evaluate population
        logging.debug("Evaluating population.")
        for g in population:
            should_discard = set()
            if g.is_evaluated():
                accuracies.append([generation, time.time() - start,
                                   g.fitness(), g.model()])
            elif str(g.model()) in seen:
                # We only get this far if the previously seen genotype has not
                # yet been evaluated. This occurs if we randomly evolve a
                # previously seen model. In this case we set the fitness to
                # -1 to disc
                g.set_fitness(-1)
                should_discard.add(g)
                continue
            
            m = NetworkTask(g.model().to_string(), dataset, 128, n_epochs=n_epochs,
                            n_modules=n_modules)
            if cuda_device_id is not None:
                r = m.run(cuda_device_id)
            else:
                r = m.run()
            accuracies.append([generation, time.time() - start,
                               r.accuracy(), g.model()])
            seen.add(str(g.model()))

        pop = set(population.offspring)
        pop.difference_update(should_discard)
        pop = list(pop)
        pop.sort(key=lambda x: x.fitness(), reverse=True)
        population.update_population(pop[:initial_size])

        logging.debug("Appending accuracies to results file.")
        append_to_file(accuracies, result_path)
        accuracies = []

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="fashion_mnist",
                        help=("Dataset to use. Can be one of mnist, "
                              "fashion_mnist (default), or cifar10."))
    parser.add_argument("--debug", action="store_true", help="Enable debug mode.")
    parser.add_argument("--population_size", type=int, default=20, help="Population size.")
    parser.add_argument("--max_layer_count", type=int, default=3,
                        help="Max number of layers per module.")
    parser.add_argument("--n_epochs", type=int, default=5, help="Number of epochs.")
    parser.add_argument("--n_modules", type=int, default=5, help="Number of modules.")
    parser.add_argument("--result_path", help="File path for results.")
    parser.add_argument("--n_generations", type=int, default=20,
                        help="Number of generations.")
    parser.add_argument("--cuda_device_id", type=int, default=0,
                        help="CUDA device ID to use.")
    return parser.parse_args()


def main():
    args = get_args()
    fmt_str = "[%(levelname)s][%(filename)s][%(funcName)s][%(lineno)d][%(message)s]"
    if args.debug:
        logging.basicConfig(format=fmt_str, level=logging.DEBUG)
        logging.debug("Using DEBUG log level.")
    else:
        logging.basicConfig(format=fmt_str, level=logging.INFO)

    if torch.cuda.is_available():
        logging.info("CUDA is available. Will attempt to use GPU device.")
    
    population = Population(args.population_size,
                            EvoBuilder(args.max_layer_count))
    run(population,
        dataset=_DATASETS[args.dataset],
        result_path=args.result_path,
        cuda_device_id=args.cuda_device_id,
        n_generations=args.n_generations,
        n_epochs=args.n_epochs,
        n_modules=args.n_modules)


if __name__ == "__main__":
    main()

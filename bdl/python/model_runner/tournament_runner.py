import logging
import pickle
import sys
import time

import numpy as np

from broker_client.broker_client import BrokerClient
from nn.classification import SimpleEvo, SimpleNN
from nn.genotype import Population
from nn.network_task import NetworkTask
from nn.data import Dataset


class TournamentRunner():

    def __init__(self, model_address, broker_client, dataset, prefix,
                 result_servicer, population, genotype_builder,
                 max_generations, n_modules, n_epochs, n_reductions,
                 result_path, log_interval, binarize, fuzz):
        self._model_address = model_address
        self._broker_client = broker_client
        self._dataset = dataset
        self._prefix = prefix
        self._result_servicer = result_servicer
        self._population = population
        self._g_builder = genotype_builder
        self._max_generations = max_generations
        self._n_modules = n_modules
        self._n_reductions = n_reductions
        self._result_path = result_path
        self._result_tracker = {}
        self._accuracies = []
        self._counter = 0
        self._start = 0.0
        self._log_interval = log_interval
        self._should_binarize = binarize
        self._should_fuzz = fuzz

    def _next_task_id(self):
        tid = "{}#{}".format(self._model_address, self._counter)
        self._counter += 1
        return tid

    def _process_results(self, generation, timeout=60*10):
        """Processes the result using tournament selection

        Args:
        generation: current generation number
        timeout: timeout to wait when popping from result queue, default: 10min.
        """

        start = time.time()
        while True:

            # Use exponential backoff, starting at 1s and resetting
            # after an interval of 60 seconds is reached.
            sleep_interval = 1
            while self._result_servicer.size() < 2:
                if sleep_interval > 60:
                    sleep_interval = 1
                time.sleep(sleep_interval)
                sleep_interval *= 2

            while self._result_servicer.size() >= 2:
                
                result = self._result_servicer.pop(timeout=timeout)
                if result is None:
                    logging.error("Hit timeout. Returning.")
                    return

                tid = result.task_id
                logging.debug("Received result for task %s" % tid)
                g = self._result_tracker[tid]
                g.set_fitness(result.result_obj.accuracy())
                logging.debug("Result loss: {}".format(result.result_obj.accuracy()))
                # Remove tid from dictionary
                out = [generation, time.time() - self.start, g.fitness(), g.model()]
                self._accuracies.append(out)
                self._result_tracker.pop(tid, None)
                self._population.add(g)

    def save_result(self):
        with open(self._result_path, "a") as f:
            for r in self._accuracies:
                f.write("{},{:.4f},{:.8f},{}\n".format(r[0], r[1], r[2], r[3]))
        self._accuracies = []

    def run(self):
        self._start = time.time()
        seen = set()
        for generation in range(self._max_generations):
            should_discard = set()
            logging.debug("Beginning generation {}".format(generation+1))

            # Generate offspring
            self._population.generate_offspring()
            logging.debug("Offspring generated")

            sent_models = 0

            # Iterate over population
            while not self._population.empty():
                g = self._population.pop()
                if g.is_evaluated() and str(g.model()) not in seen:
                    self._accuracies.append([generation, time.time() - self._start,
                                             g.fitness(), g.model()])
                    logging.debug("Skipping previously evaluated model.")
                    continue
                elif str(g.model()) in seen:
                    g.set_fitness(-1)
                    should_discard.add(g)
                    continue

                seen.add(str(g.model()))
                m = AENetworkTask(img_path=self._prefix,
                                  layers=g.model().to_string(),
                                  dataset=self._dataset,
                                  n_epochs=self._n_epochs,
                                  log_interval=self._log_interval,
                                  n_modules=self._n_modules,
                                  n_reductions=self._n_reductions,
                                  binarize=self._should_binarize,
                                  fuzz=self._should_fuzz)

                t = Task(task_id=self._next_task_id(),
                         source=self._model_address,
                         task_obj=pickle.dumps(m))

                logging.debug("Sending model {} of size {} bytes".format(g.model(),
                                                                         sys.getsizeof(g.model.layers)))
                self._broker_client.send_task(t)
                sent_models += 1
                logging.debug("Sent task %s." % t.task_id)
                self._result_tracker[t.task_id] = g

            self._process_results(generation)
            if len(self._accuracies) > 0:
                logging.debug("Saving accuracies to file.")
                self.save_results()

            # Need to modify population.
            # TODO: what is the best way to track evaluated vs outstanding
            # genotypes from the population?

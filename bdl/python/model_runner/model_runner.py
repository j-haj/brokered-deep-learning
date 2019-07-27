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

from task_service.task_service_pb2 import Task

class EvoBuilder():

    def __init__(self, max_layers):
        self.max_layers = max_layers

    def build(self):
        return SimpleEvo(self.max_layers)

class ModelRunner():

    def __init__(self, model_address, broker_client,
                 result_servicer, population, max_generations=100):
        self.model_address = model_address
        self.broker_client = broker_client
        self.result_servicer = result_servicer
        self.population = population
        self.result_tracker = {}
        self.max_generations = max_generations
        self.counter = 0
        self.accuracies = []

    def _next_task_id(self):
        tid = "{}#{}".format(self.model_address, self.counter)
        self.counter += 1
        return tid

    def _process_results(self, generation, expected_n_results, timeout):
        """Processes the results in the result_servicer's queue. Returns
        if timeout is reached.

        Args:
        expected_n_results: expected number of results
        timeout: time in seconds to wait per task
        """
        start = time.time()
        end = start + timeout * expected_n_results
        for i in range(expected_n_results):
            # Check timeout
            if time.time() > end:
                return
            result = self.result_servicer.pop(timeout=timeout*expected_n_results)
            if result is None:
                logging.debug("Hit timeout. Returning.")
                return
 
            tid = result.task_id
            logging.debug("Received result for task %s" % tid)
            g = self.result_tracker[tid]
            g.set_fitness(result.result_obj.accuracy())
            logging.debug("Result accuracy {}".format(result.result_obj.accuracy()))
            # Remove tid from dictionary
            self.accuracies.append([generation, g.fitness(), g.model()])
            self.result_tracker.pop(tid, None)

    def save_results(self, path="model_results.csv"):
        with open(path, "a") as f:
            for r in self.accuracies:
                f.write("{},{:.8f},{}\n".format(r[0], r[1], r[2]))
        self.results = []

    def run(self):
        for generation in range(self.max_generations):
            logging.debug("Beginning generation {}".format(generation+1))
            # Generate offspring
            self.population.generate_offspring()
            logging.debug("Offspring generated.")

            # Evaluate candidate networks by sending network tasks to broker
            # for distribution to workers
            sent_models = len(self.population.offspring)
            logging.debug("Sending {} models for evaluation.".format(sent_models))
            for g in self.population:
                
                if g.is_evaluated():
                    sent_models -= 1
                    self.accuracies.append([generation, g.fitness(), g.model()])
                    logging.debug("Skipping a previously evaluated model")
                    continue
                m = NetworkTask(g.model().to_string(), Dataset.FASHION_MNIST, 128, n_epochs=2)
                # Create a task
                t = Task(task_id=self._next_task_id(),
                         source=self.model_address,
                         task_obj=pickle.dumps(m))
                logging.debug("Sending model {} of size {} bytes".format(g.model(),
                                                                         sys.getsizeof(g.model().layers)))
                self.broker_client.send_task(t)
                logging.debug("Sent task %s." % t.task_id)
                self.result_tracker[t.task_id] = g

            # Process results from result servicer with a timeout of
            # 5 minutes per result
            if len(self.accuracies) > 0:
                logging.debug("Saving accuracies to file.")
                self.save_results()
            logging.debug("Waiting for results.")
            self._process_results(generation, sent_models, 5*60)

            # Update population using tournament selection
            keepers = []
            while len(keepers) < self.population.n:
                g1, g2 = np.random.choice(self.population.offspring, size=2)
                if g1.fitness() > g2.fitness():
                    keepers.append(g1)
                else:
                    keepers.append(g2)
            self.population.update_population(keepers)
            logging.info("Updated population max fitness: {:.6f}".format(
                self.population.max_fitness()))
                
                


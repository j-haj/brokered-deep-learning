import logging

import numpy as np


class Genotype(object):

    def __init__(self, model):
        self._model = model
        self._fitness = -1.0
        self._is_evaluated = False

    def fitness(self):
        """Return genotype fitness."""
        return self._fitness

    def is_evaluated(self):
        """Check if genotype has been evaluated."""
        return self._is_evaluated

    def set_fitness(self, f):
        self._is_evaluated = True
        self._fitness = f

    def mate(self, other):
        return [Genotype(m) for m in self._model.mate(other._model)]

    def model(self):
        return self._model
    
    def __repr__(self):
        return str(self._model)
    
class Population(object):

    def __init__(self, n, builder):
        self.n = n
        self.i_iter = 0
        self.population = [Genotype(builder.build()) for _ in range(n)]
        self.offspring = []

    def generate_offspring(self):
        self.offspring = [x for x in self.population]
        # Create child population
        for i in range(len(self.population)):
            idx = np.random.choice([j for j in range(len(self.population)) if j != i])
            self.offspring += self.population[i].mate(self.population[idx])
        assert len(self.offspring) >= 2 * self.n

    def update_population(self, new_population):
        self.population = new_population
        if len(new_population) != self.n:
            logging.error(("Added population with different size from start: "
                           "expected: %d got: %d") % (self.n, len(new_population)))
        self.n = len(new_population)
        self.offspring = []

    def should_be_pared(self):
        return len(self.population) > self.n

    def max_fitness(self):
        m = -1*float("inf")
        for p in self.population:
            if p.fitness() > m:
                m = p.fitness()
        return m

    def __iter__(self):
        return self

    def __next__(self):
        if self.i_iter < len(self.offspring):
            g = self.offspring[self.i_iter]
            self.i_iter += 1
            return g
        else:
            self.i_iter = 0
            raise StopIteration

        

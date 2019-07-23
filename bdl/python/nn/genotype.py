import logging

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
        return self._model.mate(other._model)

class Population(object):

    def __init__(self, n):
        self.n = n
        self.i_iter = 0
        self.population = [Genotype(EvoNet()) for _ in range(n)]
        self.offspring = []

    def generate_offspring(self):
        self.offspring = [x for x in self.population]
        # Create child population
        for i in range(self.n):
            idx = np.random.choice([j for j in range(self.n) if j != i])
            self.offspring += self.population[i].mate(self.population[idx])
        assert len(self.offspring) >= 2 * self.n

    def filter(self, filter_func):
        assert callable(filter_func)

    def __iter__(self):
        return self

    def next(self):
        if self.i_iter < len(self.offspring):
            g = self.offspring[self.i_iter]
            self.i_iter += 1
            return g
        else:
            self.i_iter = 0
            raise StopIteration

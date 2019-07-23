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
        new_pop = [x for x in self.population]
        # Create child population
        for i in range(self.n):
            idx = np.random.choice([j for j in range(self.n) if j != i])
            new_pop += self.population[i].mate(self.population[idx])
        assert len(new_pop) >= 2 * self.n

    def filter(self, new_population):
        pass

    def __iter__(self):
        return self

    def next(self):
        if self.i_iter < self.n:

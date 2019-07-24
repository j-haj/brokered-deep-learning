import numpy as np
import torch

from classification import SimpleEvo, SimpleNN


def main():
    pop = [SimpleEvo(5), SimpleEvo(5)]
    print("initial: {}".format(pop))
    for p in pop:
        p.mutate()
        print(p)

    print("Initial mutation done.")
    print(pop[0].mate(pop[1]))

    for gen in range(3):
        new_pop = [p for p in pop]
        for (i, p) in enumerate(pop):
            idx = np.random.choice([j for j in range(len(pop)) if j != i])
            new_pop += p.mate(pop[idx])
        pop = new_pop
        print("generation {}: pop: {}".format(gen, pop))

    s1 = SimpleEvo(5)
    for _ in range(5):
        s1.mutate()

    nn = SimpleNN(3, 10, s1.layers)
    nn.build()

    t = torch.rand(3, 500, 500).unsqueeze(0)
    print(t)
    print(nn.forward(t))
    
if __name__ == "__main__":
    main()

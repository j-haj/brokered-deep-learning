import numpy as np
import torch
import torch.optim as optim

from classification import SimpleEvo, SimpleNN
from data import mnist_loaders

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

    nn = SimpleNN(1, 10, s1.layers)
    nn.build()
    nn.to("cuda")
    train_loader, test_loader = mnist_loaders(64)

    optimizer = optim.SGD(nn.parameters(), lr=.1, momentum=.99)

    nn.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to("cuda"), target.to("cuda")
        optimizer.zero_grad()
        output = nn(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print("train step {}: loss: {:.6f}".format(
                batch_idx, loss.item()))


if __name__ == "__main__":
    main()

import numpy as np
import torch
import torch.optim as optim

from classification import SimpleEvo, SimpleNN
from data import mnist_loaders

def main():
    t = torch.rand(1, 500, 500).unsqueeze(0)

    s1 = SimpleEvo(5)
    for _ in range(5):
        s1.mutate()

    nn = SimpleNN(1, 10, s1.layers, 5)
    nn.build_model()

    nn(t)
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

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from classification import SimpleEvo, SimpleNN
from data import mnist_loaders



def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print("Using cuda.")

    s1 = SimpleEvo(5)
    for _ in range(5):
        s1.mutate()

    print(s1)
    nn = SimpleNN((1, 28, 28), 10, s1.layers, 2)


    nn.to(device)
    train_loader, test_loader = mnist_loaders(64)

    optimizer = optim.Adam(nn.parameters())

    nn.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
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

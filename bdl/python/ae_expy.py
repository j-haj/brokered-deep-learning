from nn.data import Dataset
from nn.autoencoder import SequentialAEEvo, SequentialAE
from nn.network_task import AENetworkTask

def main():
    s = SequentialAEEvo(max_len=2)
    for i in range(10):
        s.mutate()
        print("iteration %d" % i)
        print(s)

    task = AENetworkTask("test", s.to_string(), Dataset.FASHION_MNIST,
                         binarize=False,
                         n_modules=2, n_reductions=2)
    r = task.run()
    print("Done with loss: {}".format(1/r.accuracy()))

if __name__ == "__main__":
    main()

from nn.data import Dataset
from nn.autoencoder import SequentialAEEvo, SequentialAE
from nn.network_task import AENetworkTask

def main():
    s = SequentialAEEvo(max_len=6)
    for i in range(5):
        s.mutate()
        print("iteration %d" % i)
        print(s)

    task = AENetworkTask("test", s.to_string(), Dataset.STL10,
                         binarize=False, fuzz=False, n_epochs=10,
                         n_modules=2, n_reductions=1)
    r = task.run()
    print("Done with result: {}".format(r.accuracy()))

if __name__ == "__main__":
    main()

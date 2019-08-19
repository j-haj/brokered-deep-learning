from nn.autoencoder import SequentialAEEvo, SequentialAE

def main():
    s = SequentialAEEvo()
    for i in range(10):
        s.mutate()
        print("iteration %d" % i)
        print(s)
        
    ae = SequentialAE(s.to_string(), (1, 28, 28), 2)
    ae.build_autoencoder()

if __name__ == "__main__":
    main()

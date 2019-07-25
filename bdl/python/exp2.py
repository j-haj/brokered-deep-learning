from nn.classification import SimpleEvo, SimpleNN


class EvoBuilder():

    def __init__(self, max_layers):
        self.max_layers

    def build(self):
        return SimpleEvo(self.max_layers)

def main():
    pop = Population(10, EvoBuilder(4))

if __name__ == "__main__":
    main()

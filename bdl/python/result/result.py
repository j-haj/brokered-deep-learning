import logging

class NetworkResult(object):
    def __init__(self, accuracy, epochs=None):
        self._accuracy = accuracy
        self._epochs = epochs

    def __repr__(self):
        if self._epochs is None:
            return "{:.6f} validation accuracy".format(self._accuracy)
        return "{:.6f} validation accuracy after {} epochs".format(
            self._accuracy, self._epochs)

    def accuracy(self):
        return self._accuracy

    def epochs(self):
        return self._epochs

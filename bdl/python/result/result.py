import logging

class NetworkResult(object):
    def __init__(self, accuracy):
        self._accuracy = accuracy

    def __repr__(self):
        return "{:.6f} validation accuracy".format(self._accuracy)

    def accuracy(self):
        return self._accuracy

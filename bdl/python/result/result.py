import logging

class NetworkResult(object):
    def __init__(self, accuracy):
        self.accuracy = accuracy

    def __repr__(self):
        return "{:.6f} validation accuracy".format(self.accuracy)

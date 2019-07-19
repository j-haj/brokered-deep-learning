import logging
import time

from result import result_pb2

class EmptyTask():

    def __init__(self, n):
        self.n = n
        
    def run(self):
        logging.info("Emtpy task sleeping for %d seconds" % self.n)
        time.sleep(self.n)
        return "Done"

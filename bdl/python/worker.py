import argparse
import logging
import pickle
import time

from task_service import task_service_pb2
from task_client.task_client import TaskClient
class Worker():

    def __init__(self, broker_address):
        self.client = TaskClient(broker_address)

    def serve(self):
        try:
            while True:
                logging.debug("Requesting task")
                tid, task = self.client.request_task()
                if task is not None:
                    logging.debug("Processing task %s" % tid)
                    time.sleep(1)
                    logging.debug("Processing of task %s done." % tid)
                else:
                    logging.error("Error encountered in reconstructing task.")
        

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--broker_address", default="localhost:10001",
                        help="Address used to establish a connection with the broker.")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    return parser.parse_args()

def main():
    args = get_args()
    fmt_str = "[%(levelname)s][%(filename)s][%(funcName)s][%(lineno)d][%(message)s]"
    if args.debug:
        logging.basicConfig(format=fmt_str, level=logging.DEBUG)
        logging.debug("Using DEBUG log level.")
    else:
        logging.basicConfig(format=fmt_str, level=logging.INFO)
        
    

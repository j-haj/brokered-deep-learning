import argparse
import logging
import pickle
import sys
import time

import grpc

from task_service import task_service_pb2
from task_client.task_client import TaskClient
from result import result_pb2

class Worker():

    def __init__(self, broker_address):
        self.client = TaskClient(broker_address)

    def serve(self):
        try:
            while True:
                logging.debug("Requesting task")
                try:
                    task, runnable = self.client.request_task()
                    tid = task.task_id
                
                    if task is not None:
                        logging.debug("Processing task %s" % tid)
                        r = runnable.run()
                        logging.debug("Processing of task %s done." % tid)
                        result = result_pb2.Result(task_id=tid,
                                                   destination=task.source,
                                                   result_obj=pickle.dumps(r))
                        self.client.send_result(result)
                        logging.debug("Successfully sent result for task %s" % tid)
                    else:
                        logging.error("Error encountered in reconstructing task.")
                except grpc.RpcError as e:
                    e.details()
                    status_code = e.code()
                    logging.error("grpc error - name: {} value: {}".format(
                        status_code.name, status_code.value))

                    # TODO: change this to exponential backoff
                    time.sleep(5)
            
        except KeyboardInterrupt:
            logging.info("Shutting down worker.")

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
    w = Worker(args.broker_address)
    w.serve()

if __name__ == "__main__":
    main()

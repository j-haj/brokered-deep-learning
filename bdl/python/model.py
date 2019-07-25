import argparse
from concurrent import futures
import logging
import pickle
import queue
import time

import grpc

from empty_task import EmptyTask

from broker_client.broker_client import BrokerClient
from model_service.model_service_pb2_grpc import ModelServiceServicer
from model_service.model_service_pb2_grpc import ModelServiceStub
from model_service import model_service_pb2
from model_service import model_service_pb2_grpc
from result.result_pb2_grpc import ResultServiceServicer
from result import result_pb2
from result import result_pb2_grpc
from task_service import task_service_pb2

from result.result import NetworkResult

class Result():
    def __init__(self, task_id, result_obj):
        self.task_id = task_id
        self.result_obj = result_obj

class ResultServicer(ResultServiceServicer):

    def __init__(self):
        self.result_q = queue.Queue()
    
    def SendResult(self, request, context):
        logging.debug("Received result with task_id: {}".format(request.task_id))
        task_id = request.task_id
        try:
            obj = pickle.loads(request.result_obj)
            self.result_q.put(Result(task_id, obj))
        except pickle.UnpicklingError as e:
            logging.error("encountered error while unpickling - {}".format(e))
            return result_pb2.ResultResponse()
        return result_pb2.ResultResponse()

class TaskBuilder():
    @staticmethod
    def build_task():
        return EmptyTask(1)

class ModelServer():

    def __init__(self, model_address, broker_address):
        self.task_count = 0
        self.model_address = model_address
        self.broker_address = broker_address
        self.broker_client = BrokerClient(broker_address)
        self.broker_client.register(model_address)
        self.result_servicer = ResultServicer()
        self.server = grpc.server(futures.ThreadPoolExecutor(4))
        result_pb2_grpc.add_ResultServiceServicer_to_server(self.result_servicer,
                                                            self.server)
        self.server.add_insecure_port(model_address)


    def generate_task(self):
        t = TaskBuilder.build_task()
        tid = "{}#{}".format(self.model_address, self.task_count)
        self.task_count += 1
        return task_service_pb2.Task(task_id=tid, source=self.model_address,
                                     task_obj=pickle.dumps(t))

    def serve(self):
        self.server.start()
        logging.info("Listening on %s" % self.model_address)
        try:
            while True:
                self.process_results()
                time.sleep(1)
                logging.debug("Generating new task")
                self.broker_client.send_task(self.generate_task())
                logging.debug("Task sent.")
        except KeyboardInterrupt:
            logging.info("Shutting down model server.")
            self.server.stop(0)

    def process_results(self):
        logging.debug("Checking result_servicer result queue.")
        while not self.result_servicer.result_q.empty():
            r = self.result_servicer.result_q.get()
            logging.info("Processing result: {}".format(r))

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_address", default="localhost:20000",
                        help="Address the model server is listenting to.")
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

    server = ModelServer(model_address=args.model_address,
                         broker_address=args.broker_address)
    server.serve()
    
if __name__ == "__main__":
    main()

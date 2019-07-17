import logging
import pickle

import grpc

from task_service import task_service_pb2
from task_service import task_service_pb2_grpc

class TaskClient():

    def __init__(self, broker_address):
        self.broker_address = broker_address
        self.channel = grpc.insercure_channel(broker_address)
        self.stub = task_service_pb2_grpc.TaskServiceStub(self.channel)

    def request_task(self):
        request = task_service_pb2.TaskRequest()
        task = self.stub.RequestTask(request)
        try:
            task_obj = pickle.loads(task.task_obj)
            return (task.task_id, task_objc)
        except pickle.UnpicklingError as e:
            logging.error("encountered error while unpickling - {}".format(e))
            return (task.task_id, None)
            


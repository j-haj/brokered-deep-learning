import logging
import pickle

import grpc

from task_service import task_service_pb2
from task_service import task_service_pb2_grpc
from result import result_pb2
from result import result_pb2_grpc


class TaskClient():

    def __init__(self, broker_address):
        self.broker_address = broker_address
        options = [("grpc.max_receive_message_length", 500 * 1024 * 1024)]
        self.channel = grpc.insecure_channel(broker_address, options=options)
        self.task_stub = task_service_pb2_grpc.TaskServiceStub(self.channel)
        self.result_stub = result_pb2_grpc.ResultServiceStub(self.channel)

    def request_task(self):
        request = task_service_pb2.TaskRequest()
        task = self.task_stub.RequestTask(request)
        try:
            task_obj = pickle.loads(task.task_obj)
            return (task, task_obj)
        except pickle.UnpicklingError as e:
            logging.error("encountered error while unpickling - {}".format(e))
            return (task, None)

    def send_result(self, result):
        logging.debug("Sending task %s result to %s." % (result.task_id,
                                                         result.destination))
        self.result_stub.SendResult(result)
        return
            


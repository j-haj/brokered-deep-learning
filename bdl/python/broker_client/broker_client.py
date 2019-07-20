import logging
import pickle

import grpc

from model_service import model_service_pb2
from model_service import model_service_pb2_grpc

class BrokerClient():

    def __init__(self, broker_address):
        self.broker_address = broker_address
        self.channel = grpc.insecure_channel(broker_address)
        self.client = model_service_pb2_grpc.ModelServiceStub(self.channel)

    def register(self, address):
        req = model_service_pb2.RegistrationRequest(address=address)
        resp = self.client.RegisterModel(req)
        if resp.success:
            logging.debug("Registered with broker %s" % address)
        else:
            logging.error("Failed to register with broker %s" % address)

    def send_task(self, task):
        self.client.SendTask(task)

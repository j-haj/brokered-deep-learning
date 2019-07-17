import argparse
from concurrent import futures
import logging
import pickle

import grpc
from model_service.model_service_pb2_grpc import ModelServiceServicer
from model_serice import model_serice_pb2
from model_serice import model_service_pb2_grpc
from result.result_pb2_grpc import ResultServiceServicer
from result import result_pb2
from result import result_pb2_grpc
from task_service import task_service_pb2

class ModelServer(ResultServiceServicer):

    def __init__(self, address):
        pass
    
    def SendResult(self, request, context):
        pass

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("address", default="localhost:20000",
                        help="Address the model server is listenting to.")

    return parser.parse_args()

    
def main():
    pass

if __name__ == "__main__":
    main()

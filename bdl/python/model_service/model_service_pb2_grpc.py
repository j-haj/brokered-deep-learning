# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
import grpc

from model_service import model_service_pb2 as model__service_dot_model__service__pb2
from task_service import task_service_pb2 as task__service_dot_task__service__pb2


class ModelServiceStub(object):
  # missing associated documentation comment in .proto file
  pass

  def __init__(self, channel):
    """Constructor.

    Args:
      channel: A grpc.Channel.
    """
    self.RegisterModel = channel.unary_unary(
        '/model_service.ModelService/RegisterModel',
        request_serializer=model__service_dot_model__service__pb2.RegistrationRequest.SerializeToString,
        response_deserializer=model__service_dot_model__service__pb2.RegistrationResponse.FromString,
        )
    self.SendTask = channel.unary_unary(
        '/model_service.ModelService/SendTask',
        request_serializer=task__service_dot_task__service__pb2.Task.SerializeToString,
        response_deserializer=model__service_dot_model__service__pb2.SendResponse.FromString,
        )


class ModelServiceServicer(object):
  # missing associated documentation comment in .proto file
  pass

  def RegisterModel(self, request, context):
    """Register is called by a model to a broker.
    """
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def SendTask(self, request, context):
    """SendTask is called by the model to send a task to the broker.
    """
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')


def add_ModelServiceServicer_to_server(servicer, server):
  rpc_method_handlers = {
      'RegisterModel': grpc.unary_unary_rpc_method_handler(
          servicer.RegisterModel,
          request_deserializer=model__service_dot_model__service__pb2.RegistrationRequest.FromString,
          response_serializer=model__service_dot_model__service__pb2.RegistrationResponse.SerializeToString,
      ),
      'SendTask': grpc.unary_unary_rpc_method_handler(
          servicer.SendTask,
          request_deserializer=task__service_dot_task__service__pb2.Task.FromString,
          response_serializer=model__service_dot_model__service__pb2.SendResponse.SerializeToString,
      ),
  }
  generic_handler = grpc.method_handlers_generic_handler(
      'model_service.ModelService', rpc_method_handlers)
  server.add_generic_rpc_handlers((generic_handler,))

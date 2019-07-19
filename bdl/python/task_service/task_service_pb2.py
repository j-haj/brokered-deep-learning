# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: task_service/task_service.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from result import result_pb2 as result_dot_result__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='task_service/task_service.proto',
  package='task_service',
  syntax='proto3',
  serialized_options=_b('Z!github.com/j-haj/bdl/task_service'),
  serialized_pb=_b('\n\x1ftask_service/task_service.proto\x12\x0ctask_service\x1a\x13result/result.proto\"9\n\x04Task\x12\x0f\n\x07task_id\x18\x01 \x01(\t\x12\x0e\n\x06source\x18\x02 \x01(\t\x12\x10\n\x08task_obj\x18\x03 \x01(\x0c\"\r\n\x0bTaskRequest\"\x10\n\x0eResultResponse2\x8b\x01\n\x0bTaskService\x12>\n\x0bRequestTask\x12\x19.task_service.TaskRequest\x1a\x12.task_service.Task\"\x00\x12<\n\nSendResult\x12\x0e.result.Result\x1a\x1c.task_service.ResultResponse\"\x00\x42#Z!github.com/j-haj/bdl/task_serviceb\x06proto3')
  ,
  dependencies=[result_dot_result__pb2.DESCRIPTOR,])




_TASK = _descriptor.Descriptor(
  name='Task',
  full_name='task_service.Task',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='task_id', full_name='task_service.Task.task_id', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='source', full_name='task_service.Task.source', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='task_obj', full_name='task_service.Task.task_obj', index=2,
      number=3, type=12, cpp_type=9, label=1,
      has_default_value=False, default_value=_b(""),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=70,
  serialized_end=127,
)


_TASKREQUEST = _descriptor.Descriptor(
  name='TaskRequest',
  full_name='task_service.TaskRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=129,
  serialized_end=142,
)


_RESULTRESPONSE = _descriptor.Descriptor(
  name='ResultResponse',
  full_name='task_service.ResultResponse',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=144,
  serialized_end=160,
)

DESCRIPTOR.message_types_by_name['Task'] = _TASK
DESCRIPTOR.message_types_by_name['TaskRequest'] = _TASKREQUEST
DESCRIPTOR.message_types_by_name['ResultResponse'] = _RESULTRESPONSE
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

Task = _reflection.GeneratedProtocolMessageType('Task', (_message.Message,), dict(
  DESCRIPTOR = _TASK,
  __module__ = 'task_service.task_service_pb2'
  # @@protoc_insertion_point(class_scope:task_service.Task)
  ))
_sym_db.RegisterMessage(Task)

TaskRequest = _reflection.GeneratedProtocolMessageType('TaskRequest', (_message.Message,), dict(
  DESCRIPTOR = _TASKREQUEST,
  __module__ = 'task_service.task_service_pb2'
  # @@protoc_insertion_point(class_scope:task_service.TaskRequest)
  ))
_sym_db.RegisterMessage(TaskRequest)

ResultResponse = _reflection.GeneratedProtocolMessageType('ResultResponse', (_message.Message,), dict(
  DESCRIPTOR = _RESULTRESPONSE,
  __module__ = 'task_service.task_service_pb2'
  # @@protoc_insertion_point(class_scope:task_service.ResultResponse)
  ))
_sym_db.RegisterMessage(ResultResponse)


DESCRIPTOR._options = None

_TASKSERVICE = _descriptor.ServiceDescriptor(
  name='TaskService',
  full_name='task_service.TaskService',
  file=DESCRIPTOR,
  index=0,
  serialized_options=None,
  serialized_start=163,
  serialized_end=302,
  methods=[
  _descriptor.MethodDescriptor(
    name='RequestTask',
    full_name='task_service.TaskService.RequestTask',
    index=0,
    containing_service=None,
    input_type=_TASKREQUEST,
    output_type=_TASK,
    serialized_options=None,
  ),
  _descriptor.MethodDescriptor(
    name='SendResult',
    full_name='task_service.TaskService.SendResult',
    index=1,
    containing_service=None,
    input_type=result_dot_result__pb2._RESULT,
    output_type=_RESULTRESPONSE,
    serialized_options=None,
  ),
])
_sym_db.RegisterServiceDescriptor(_TASKSERVICE)

DESCRIPTOR.services_by_name['TaskService'] = _TASKSERVICE

# @@protoc_insertion_point(module_scope)
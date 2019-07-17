#!/bin/bash

PYTHON=/usr/local/bin/python3

$PYTHON -m grpc_tools.protoc -I../ --python_out=. --grpc_python_out=. ../task_service/task_service.proto
$PYTHON -m grpc_tools.protoc -I../ --python_out=. --grpc_python_out=. ../result/result.proto
$PYTHON -m grpc_tools.protoc -I../ --python_out=. --grpc_python_out=. ../model_service/model_service.proto


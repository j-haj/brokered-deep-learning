#!/bin/bash

protoc -I. task_service/task_service.proto --go_out=plugins=grpc,paths=source_relative:.
protoc -I. broker_comm/broker_comm.proto --go_out=plugins=grpc,paths=source_relative:.
protoc -I. nameservice/nameservice.proto --go_out=plugins=grpc,paths=source_relative:.
protoc -I. heartbeat/heartbeat.proto --go_out=plugins=grpc,paths=source_relative:.
protoc -I. result/result.proto --go_out=plugins=grpc,paths=source_relative:.
protoc -I. model_service/model_service.proto --go_out=plugins=grpc,paths=source_relative:.

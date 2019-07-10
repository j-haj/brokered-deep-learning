#!/bin/bash

protoc -I. nameservice/nameservice.proto --go_out=plugins=grpc,paths=source_relative:.

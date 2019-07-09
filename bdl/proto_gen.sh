#!/bin/bash

protoc -I. nameserver/nameservice.proto --go_out=plugins=grpc,paths=source_relative:.

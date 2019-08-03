#!/bin/bash
# Set out files
N=4
NS_OUT="ns$N.out"
B_OUT="broker$N.out"
M_OUT="model$N.out"
RESULTS_PATH="results_$N.csv"
printf "Saving results to $RESULTS_PATH"

# Server addresses
NAMESERVER_ADDR="localhost:10003"
BROKER_ADDR="localhost:10004"
MODEL_ADDR="localhost:20003"

# Start nameserver
nohup go run nameserver/nameserver.go -nameserver_address=$NAMESERVER_ADDR > $NS_OUT &
NS_PID=$!

sleep .5

# Start broker
nohup go run broker/broker.go \
      -nameserver_address=$NAMESERVER_ADDR \
      -broker_address=$BROKER_ADDR \
      -n_attempted_connections=0 > $B_OUT &
BROKER_PID=$!

sleep .5

# Start workers
nohup python3 python/worker.py \
      --debug \
      --broker_address=$BROKER_ADDR \
      --cuda_device_id=2 > w1-$N.out &
W1_PID=$!

nohup python3 python/worker.py \
      --debug \
      --broker_address=$BROKER_ADDR \
      --cuda_device_id=3 > w2-$N.out &
W2_PID=$!

nohup python3 python/worker.py \
      --debug \
      --broker_address=$BROKER_ADDR \
      --cuda_device_id=4 > w3-$N.out &
W3_PID=$!

nohup python3 python/worker.py \
      --debug \
      --broker_address=$BROKER_ADDR \
      --cuda_device_id=5 > w4-$N.out &
W4_PID=$!

echo "$N workers started. Starting model."

# Start model
nohup python3 python/model.py \
      --model_address=$MODEL_ADDR \
      --broker_address=$BROKER_ADDR \
      --dataset=cifar10 \
      --epochs=10 \
      --population_size=10 \
      --max_layer_size=5 \
      --n_modules=3 \
      --result_path=$RESULTS_PATH \
      --debug > $M_OUT &

MODEL_PID=$!

echo "PIDs:"
printf "\tNS: $NS_PID\n\tBROKER: $BROKER_PID\n\tMODEL: $MODEL_PID"


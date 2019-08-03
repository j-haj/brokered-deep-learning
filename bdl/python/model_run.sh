#!/bin/bash

python3 model.py \
	--model_address=localhost:20002 \
	--broker_address=localhost:10004 \
	--dataset=cifar10 \
	--epochs=10 \
	--population_size=10 \
	--max_layer_size=5 \
	--n_modules=3 \
	--result_path="results_4workers.csv" \
	--debug


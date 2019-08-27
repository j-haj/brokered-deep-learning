#!/bin/bash

python3 model.py \
	--model_address=localhost:20002 \
	--broker_address=localhost:10004 \
	--dataset=mnist \
	--epochs=2 \
	--population_size=20 \
	--max_layer_size=5 \
	--n_modules=2 \
	--result_path="results_4workers.csv" \
	--debug


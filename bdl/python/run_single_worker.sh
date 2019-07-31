#!/bin/bash

python3 single_worker.py \
	--dataset=cifar10 \
	--population_size=10 \
	--max_layer_count=5 \
	--n_epochs=10 \
	--n_modules=3 \


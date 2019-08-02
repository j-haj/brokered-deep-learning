#!/bin/bash

python3 model.py \
	--dataset=cifar10 \
	--epochs=10 \
	--population_size=10 \
	--max_layer_size=5 \
	--n_modules=3 \
	--debug


#!/bin/bash

nohup python3 single_worker.py \
      --debug \
      --dataset=mnist \
      --population_size=2 \
      --max_layer_count=5 \
      --n_epochs=2 \
      --n_modules=3 \
      --result_path="single_results.csv" \
      --n_generations=20 \
      --cuda_device_id=0 > single.out &


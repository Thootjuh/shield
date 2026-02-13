#!/usr/bin/env bash

NUM_THREADS=10
NUM_ITERATIONS=10
NUM_ITERATION_PER_THREAD=$((NUM_ITERATIONS/NUM_THREADS)) # make sure its divisible, otherwise it will be floored.

SEED=15077084497497197234

echo "INFO: Running experiments"
echo "INFO: Running $NUM_ITERATION_PER_THREAD iterations in $NUM_THREADS threads for a total of $NUM_ITERATIONS"

for env in random_mdps wet_chicken frozen_lake pacman
do
    echo "Running experiments on $env"
    python3 run_experiments.py ${env}_shield.ini experiment_results $SEED $NUM_THREADS $NUM_ITERATION_PER_THREAD
done

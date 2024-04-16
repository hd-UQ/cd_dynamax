#!/bin/bash

# Define the environments and the parameters
environments=("hduq_nodynamax" "hduq_nodynamax_GPU")
sequence_counts=(10 100 1000 10000)

i=1
T=1
b=1
# -i 3: run 3 iterations of the same code block for more robust timing estimates
# -T 10: This defines the length of the simulated trajectory in model time units
# -N: The batch number (We simulate N trajectories (each of length T) in parallel)
# -b: The batch size in SGD

# source the conda.sh script to activate conda
source ~/anaconda3/etc/profile.d/conda.sh

# Loop over each environment
for env in "${environments[@]}"; do
    echo "Activating environment: $env"
    conda activate "$env"
    
    # Loop over each sequence count
    for N in "${sequence_counts[@]}"; do
        echo "Running timer.py in environment $env with N=$N"
        python timer_sgd.py -i "$i" -T "$T" -N "$N" -b "$b"
    done
    
    echo "Deactivating environment: $env"
    conda deactivate
done

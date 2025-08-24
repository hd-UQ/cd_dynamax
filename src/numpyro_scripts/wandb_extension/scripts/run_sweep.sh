#!/bin/bash
set -euo pipefail

SWEEP_CONFIG=$1
N_AGENTS=$2 # number of parallel agents to launch
COUNTS_PER_AGENT=$3 # number of runs per agent

TEXT=$(wandb sweep "$SWEEP_CONFIG" 2>&1)
echo "$TEXT"

SWEEP_ID=$(python /Users/levinema/Projects/research/cd_dynamax_private/src/numpyro_scripts/wandb_extension/scripts/parse_wandb_sweep.py <<< "$TEXT")
echo "Parsed sweep ID: $SWEEP_ID"
echo "Launching $N_AGENTS agents for sweep $SWEEP_ID"

# Launch N agents in parallel
for i in $(seq 1 $N_AGENTS); do
  echo "Starting agent $i..."
  wandb agent "$SWEEP_ID" --count "$COUNTS_PER_AGENT" &
done

wait

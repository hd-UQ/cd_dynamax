#!/usr/bin/env python3
import sys
import re

lines = sys.stdin.read().strip().splitlines()
sweep_id = None

for line in lines:
    if "wandb agent" in line:
        # capture everything after "wandb agent "
        match = re.search(r'wandb agent (\S+)', line)
        if match:
            sweep_id = match.group(1)
            break

if not sweep_id:
    print("ERROR: could not parse sweep ID", file=sys.stderr)
    sys.exit(1)

print(sweep_id)

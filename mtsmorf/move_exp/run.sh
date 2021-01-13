#!/bin/bash

#### Bash script to run move_spectral_power.py script for all subjects.

declare -a subjects=(
    "efri02" 
    "efri06" 
    "efri07" 
    # "efri09"  # Too few samples
    # "efri10"  # Unequal data size vs label size
    "efri13" 
    "efri14" 
    "efri15" 
    "efri18" 
    "efri20" 
    "efri26" 
    )

for subject in "${subjects[@]}"
do
    echo "Running $subject..."
    # python3 move_spectral_power.py $subject
    python3 ./move_exp/experiments.py $subject -experiment baseline
    # python3 ./move_exp/experiments.py $subject -experiment onset
    # python3 ./move_exp/experiments.py $subject -experiment shuffle
    # python3 ./move_exp/move_spectral_power.py $subject --replot-signals True --feat-importances False --rerun-fit False
    echo "Done with $subject..."
done
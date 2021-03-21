#!/bin/sh

#### Bash script to run experiments for all subjects.

SCRIPTPATH="$(
  cd "$(dirname "$0")"
  pwd -P
)"

declare -a subjects=(
  # "efri02"
  # "efri06"
  # "efri07"
  # "efri09"  # Too few samples
  # "efri10"  # Unequal data size vs label size
  # "efri13"
  "efri14"
  "efri15"
  "efri18"
  "efri20"
  "efri25"
  "efri26"
)

for subject in "${subjects[@]}"; do
  echo "Running $subject..."
  # python3 ./move_exp/experiments.py $subject -experiment shuffle
  # python3 ./move_exp/movement_onset_experiment.py $subject
  # python3 ./move_exp/experiments.py $subject -experiment baseline
  # python3 ./move_exp/experiments.py $subject -experiment frequency_bands
  # python3 ./move_exp/experiments.py $subject -experiment trial_specific_time_window_time
  # python3 ./move_exp/experiments.py $subject -experiment trial_specific_time_window_freq
  # python3 ./move_exp/experiments.py $subject -experiment plot_event_durations
  # python3 ./move_exp/experiments.py $subject -experiment plot_event_onsets
  # python3 ./move_exp/move_spectral_power.py $subject --replot-signals True --feat-importances False --rerun-fit False
  echo "Starting Decode Movement"
  chmod +x "$SCRIPTPATH/decoding_movement.py"
  python "$SCRIPTPATH/decoding_movement.py" $subject
  echo "Done with Decode Movement"

  echo "Starting Decode Directionality"
  chmod +x "$SCRIPTPATH/decoding_directionality.py"
  python "$SCRIPTPATH/decoding_directionality.py" $subject
  echo "Done with Decode Directionality"

  echo "Starting Speed Instruction"
  chmod +x "$SCRIPTPATH/speed_instruction.py"
  python "$SCRIPTPATH/speed_instruction.py" $subject
  echo "Done with Speed Instruction"

  echo "Done with $subject..."
done

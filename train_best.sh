#!/bin/bash

configs=( "w fnn" "w cnn" "w tcn" "w lstm" "w bilstm" ) #"pv fnn" "pv cnn" "pv tcn" "pv lstm" "pv bilstm" )



start_screen_session() {
    local d=$1
    local m=$2
    local b='best'
    local session_name="${d}_${m}_${b}"
    local log_file="logs/output_${session_name}.log"

    screen -dmS "$session_name" bash -c "source fedsim/bin/activate; python sim.py -d $d -m $m -e > $log_file 2>&1"
}

mkdir -p logs

for config in "${configs[@]}"; do
    IFS=' ' read -r d m <<< "$config"
    echo "Starting screen session ${d}_${m}_${b} with configurations -d $d -m $m"
    start_screen_session "$d" "$m"
done

echo "All screen sessions started."
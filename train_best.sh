#!/bin/bash

configs=( "w fnn" "w lstm" "w tcn" "w xgb" " w cnn" "w bilstm" "pv fnn" "pv lstm" "pv tcn" "pv xgb" " pv cnn" "pv bilstm")



start_screen_session() {
    local d=$1
    local m=$2
    local session_name="${d}_${m}_best"
    local log_file="logs/output_${session_name}.log"

    screen -dmS "$session_name" bash -c "source fedsim/bin/activate; python sim.py -d $d -m $m -e > $log_file 2>&1"
}

mkdir -p logs

for config in "${configs[@]}"; do
    IFS=' ' read -r d m <<< "$config"
    echo "Starting screen session ${d}_${m}_best with configurations -d $d -m $m"
    start_screen_session "$d" "$m"
done

echo "All screen sessions started."
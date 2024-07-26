#!/bin/bash

screen -ls | grep -E '[0-9]+\.' | awk '{print $1}' | while read -r session; do
    echo "Closing screen session $session"
    screen -S "$session" -X quit
done

echo "All screen sessions closed."
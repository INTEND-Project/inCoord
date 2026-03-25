#!/bin/bash

pkill -9 -f locust
pkill -9 python3
PID=$(lsof -t -i:5000)

if [ -n "$PID" ]; then
    echo "Killing process on port 5000 (PID: $PID)"
    kill "$PID"
fi

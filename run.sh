#!/bin/bash
bash stop.sh

ROUNDS=1
for ((i=1; i<=ROUNDS; i++)); do
    python ../reset.py
    rm timing1.json
    curl -X POST http://172.16.0.1:32663/clear_data
    #start locust users
    locust -f ./locust/locustfile.py --headless -u 10 -r 1 --run-time "3600"&
    PID=$!
    echo "Locust started with PID $PID"
    python coordinator.py --lower 2000 --upper 3000 --wandb_name "New Coordinator tests" --strategy "reactive" --mode "no_coord" &
    PID2=$!
    echo "Coordination started with PID $PID2"
    sleep 3600
    locust -f ./locust/locustfile.py --headless -u 20 -r 1 --run-time "3600"&
    sleep 3600
    locust -f ./locust/locustfile.py --headless -u 10 -r 1 --run-time "3600"&
    sleep 3600
    locust -f ./locust/locustfile.py --headless -u 5 -r 1 --run-time "3600"&
    sleep 3600
    bash stop.sh
    sleep 30
    bash stop.sh
done

for ((i=1; i<=ROUNDS; i++)); do
    python ../reset.py
    rm timing1.json
    curl -X POST http://172.16.0.1:32663/clear_data
    #start locust users
    locust -f ./locust/locustfile.py --headless -u 10 -r 1 --run-time "3600"&
    PID=$!
    echo "Locust started with PID $PID"
    python coordinator.py --lower 2000 --upper 3000 --wandb_name "New Coordinator tests" --strategy "PPO" --mode "no_coord" &
    PID2=$!
    echo "Coordination started with PID $PID2"
    sleep 3600
    locust -f ./locust/locustfile.py --headless -u 20 -r 1 --run-time "3600"&
    sleep 3600
    locust -f ./locust/locustfile.py --headless -u 10 -r 1 --run-time "3600"&
    sleep 3600
    locust -f ./locust/locustfile.py --headless -u 5 -r 1 --run-time "3600"&
    sleep 3600
    bash stop.sh
    sleep 30
    bash stop.sh
done
for ((i=1; i<=ROUNDS; i++)); do
    python ../reset.py
    rm timing1.json
    curl -X POST http://172.16.0.1:32663/clear_data
    #start locust users
    locust -f ./locust/locustfile.py --headless -u 10 -r 1 --run-time "3600"&
    PID=$!
    echo "Locust started with PID $PID"
    python coordinator.py --lower 2000 --upper 3000 --wandb_name "New Coordinator tests" --strategy "reactive" --mode "TD3_context32500010" &
    PID2=$!
    echo "Coordination started with PID $PID2"
    sleep 3600
    locust -f ./locust/locustfile.py --headless -u 20 -r 1 --run-time "3600"&
    sleep 3600
    locust -f ./locust/locustfile.py --headless -u 10 -r 1 --run-time "3600"&
    sleep 3600
    locust -f ./locust/locustfile.py --headless -u 5 -r 1 --run-time "3600"&
    sleep 3600
    bash stop.sh
    sleep 30
    bash stop.sh
done
for ((i=1; i<=ROUNDS; i++)); do
    python ../reset.py
    rm timing1.json
    curl -X POST http://172.16.0.1:32663/clear_data
    #start locust users
    locust -f ./locust/locustfile.py --headless -u 10 -r 1 --run-time "3600"&
    PID=$!
    echo "Locust started with PID $PID"
    python coordinator.py --lower 2000 --upper 3000 --wandb_name "New Coordinator tests" --strategy "PPO" --mode "TD3_context32500010" &
    PID2=$!
    echo "Coordination started with PID $PID2"
    sleep 3600
    locust -f ./locust/locustfile.py --headless -u 20 -r 1 --run-time "3600"&
    sleep 3600
    locust -f ./locust/locustfile.py --headless -u 10 -r 1 --run-time "3600"&
    sleep 3600
    locust -f ./locust/locustfile.py --headless -u 5 -r 1 --run-time "3600"&
    sleep 3600
    bash stop.sh
    sleep 30
    bash stop.sh
done
for ((i=1; i<=ROUNDS; i++)); do
    python ../reset.py
    rm timing1.json
    curl -X POST http://172.16.0.1:32663/clear_data
    #start locust users
    locust -f ./locust/locustfile.py --headless -u 10 -r 1 --run-time "3600"&
    PID=$!
    echo "Locust started with PID $PID"
    python coordinator.py --lower 2000 --upper 3000 --wandb_name "New Coordinator tests" --strategy "PPO" --mode "reactive" &
    PID2=$!
    echo "Coordination started with PID $PID2"
    sleep 3600
    locust -f ./locust/locustfile.py --headless -u 20 -r 1 --run-time "3600"&
    sleep 3600
    locust -f ./locust/locustfile.py --headless -u 10 -r 1 --run-time "3600"&
    sleep 3600
    locust -f ./locust/locustfile.py --headless -u 5 -r 1 --run-time "3600"&
    sleep 3600
    bash stop.sh
    sleep 30
    bash stop.sh
done
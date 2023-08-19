#!/bin/bash

echo "Arguments passed: $1 $2"

PATTERN="$1"
output="flower_clients_monitor.txt"
echo "Starting monitoring..." > $output
expected_pid_count="$2"
echo "$expected_pid_count"
# Variables to accumulate data
iterations=0
total_sent=0
total_cpu=0
total_ram=0

# Wait until a process with the pattern starts
while true; do
    if ps aux | grep "$PATTERN" | grep -v grep > /dev/null; then
        echo "Detected processes with the pattern $PATTERN" >> $output
        break
    fi
    sleep 5
done

# Start monitoring as long as processes with the pattern exist
while true; do
    process_count=$(ps aux | grep "$PATTERN" | grep -v grep | grep -v "$0" | wc -l)
    echo "$process_count"
    cpus_used=$(ps -o psr= -p $(pgrep -f "$PATTERN" | tr '\n' ',' | sed 's/,$//'))
    # Exit loop if no processes with the pattern are found
    if [ "$process_count" -eq 0 ]; then
        break
    fi

    # Collect and process CPU and RAM data
    read current_cpu current_ram < <(ps aux | grep "$PATTERN" | grep -v grep | awk '{
       total_cpu += $3; total_ram += $4; } END { print total_cpu, total_ram }')



    total_cpu=$(echo "$total_cpu + $current_cpu" | bc)
    total_ram=$(echo "$total_ram + $current_ram" | bc)

    nethogs -c 2 -t > nethogs_debug.txt
    # Collect and process network data
    current_sent=$(cat nethogs_debug.txt | grep -E "python|python3" | awk -F"\t" '{ total_sent += $2 } END { print total_sent }')
    [ -z "$current_sent" ] && current_sent=0
    total_sent=$(echo "$total_sent + $current_sent" | bc)
    echo "Debug: Iterations: $iterations, Current CPU: $current_cpu, Total CPU: $total_cpu, Current RAM: $current_ram, Total RAM: $total_ram, SENT: $current_sent, pid count: $expected_pid_count"
    echo "Current CPU: $current_cpu, RAM: $current_ram, Sent: $current_sent" >> $output
    iterations=$((iterations + 1))
done

echo "Debug Before Average Calculation: CPU: $total_cpu, RAM: $total_ram, SENT: $total_sent, Iterations: $iterations, Expected PID Count: $expected_pid_count"

avg_cpu_value=$(echo "scale=2; ($total_cpu / ($iterations * $expected_pid_count))" | bc)
avg_ram_value=$(echo "scale=2; ($total_ram / ($iterations * $expected_pid_count))" | bc)
avg_sent_value=$(echo "scale=2; $total_sent / $expected_pid_count" | bc)


date="$(date +'%Y-%m-%d %H:%M:%S')"
echo "$date: Average CPU: $avg_cpu_value%, Average RAM: $avg_ram_value% based on expected count of $expected_pid_count. Average packets sent per process: $avg_sent_value KB." >> $output
echo "$date: Average CPU: $avg_cpu_value%, Average RAM: $avg_ram_value% based on expected count of $expected_pid_count. Average packets sent per process: $avg_sent_value KB."
echo "CPUs used: $cpus_used"

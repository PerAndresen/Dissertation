#! /bin/bash

for i in {1..10}

do 
    python3 client.py --cid $i --num_clients 10 --server_address "192.168.4.25:8080" &
done
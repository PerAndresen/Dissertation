#! /bin/bash

for i in {1..10}

do 
    python3 sa_client.py --cid $i --num_clients 2 --server_address "192.168.4.25:8080" &
done
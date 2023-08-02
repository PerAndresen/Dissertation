#! /bin/bash

for i in {0..1}

do 
    python3 clienten.py --cid $i --num_clients 10 --server_address "192.168.4.25:8080" &
done
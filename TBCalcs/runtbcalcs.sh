#!/bin/bash

for i in {4..20}
do
    echo "Running simulation with separation size $i"
    python ldos.py $i ./output/sep$i 
done

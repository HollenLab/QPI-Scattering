#!/bin/bash

for i in {1..9}
do
    echo "Running simulation with separation size 0. $i"
    python single_filter_atr.py 0.$i
    echo "Running simulation with separation size 1. $i"
    python single_filter_atr.py 1.$i

done

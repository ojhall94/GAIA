#!/bin/bash

TEMPS = {-50., 60., 10.}

for i in {-50..50..10}; do
     python test.py "tempscale" $i
done
echo 'Complete!'

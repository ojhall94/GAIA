#!/bin/bash

#Arguments are:
#[1]: Type
#[2]: Iterations
#[3]: corrections [None, RC]
#[4]: band [J, H, K, GAIA]
#[5]: tempdiff

#Tempdiff in K, no correction
for i in {-50..50..10}; do
     python bash_stan.py 'astero' 5000 'None' 'K' $i
done

#Temp diff in K, with correction
for i in {-50..50..10}; do
     python bash_stan.py 'astero' 5000 'RC' 'K' $i
done

# #Tempdiff in GAIA, no correction
# for i in {-50..50..10}; do
#      python bash_stan.py 'astero' 5000 'None' 'GAIA' $i
# done
#
# #Temp diff in GAIA, with correction
# for i in {-50..50..10}; do
#      python bash_stan.py 'astero' 5000 'RC' 'GAIA' $i
# done

# #Tempdiff in J, no correction
# for i in {-50..50..10}; do
#      python bash_stan.py 10000 'None' 'J' $i
# done
#
# #Temp diff in J, with correction
# for i in {-50..50..10}; do
#      python bash_stan.py 10000 'RC' 'J' $i
# done
#
# #Tempdiff in H, no correction
# for i in {-50..50..10}; do
#      python bash_stan.py 10000 'None' 'H' $i
# done
#
# #Temp diff in H, with correction
# for i in {-50..50..10}; do
#      python bash_stan.py 10000 'RC' 'H' $i
# done
#

echo 'Complete!'

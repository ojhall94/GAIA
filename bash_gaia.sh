#!/bin/bash

#Arguments are:
#[0]: Type [gaia, astero]
#[1]: Iterations
#[2]: corrections [None, RC]
#[3]: band [J, H, K, GAIA]
#[4]: tempdiff

#Tempdiff in K, no correction
for i in {-50..50..10}; do
     python bash_stan.py 'gaia' 1000 'None' 'K' $i
done
cp astrostan.pkl ../Output

#Temp diff in K, with correction
for i in {-50..50..10}; do
     python bash_stan.py 'gaia' 1000 'RC' 'K' $i
done
cp astrostan.pkl ../Output


# #Tempdiff in GAIA, no correction
# for i in {-50..50..10}; do
#      python bash_stan.py 5000 'None' 'GAIA' $i
# done
#
# #Temp diff in GAIA, with correction
# for i in {-50..50..10}; do
#      python bash_stan.py 5000 'RC' 'GAIA' $i
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

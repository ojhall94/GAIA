#!/bin/bash

# Run our PyStan model on some data
#
# positional arguments:
#   {astero,gaia}  Choice of PyStan model.
#   iters          Number of MCMC iterations in PyStan.
#   {None,RC}      Choice of corrections to the seismic scaling relations.
#   {K,J,H,GAIA}   Choice of photometric passband.
#   tempdiff       Perturbation to the temperature values in K
#   bclabel        Temp arg: nn: no prop; lt prop logg and teff; nt prop t only.

#
# optional arguments:
#   -h, --help     show this help message and exit
#   --testing, -t  Turn on to output results to a test_build folder
#   --update, -u   Turn on to update the PyStan model you choose to run

##################################PROPAGATE TEFF AND LOGG

#Tempdiff in K, no correction
for i in {-50..50..10}; do
     python bash_stan.py 'astero' 10000 'None' 'K' $i 'lt'
done

#Temp diff in K, with correction
for i in {-50..50..10}; do
     python bash_stan.py 'astero' 10000 'RC' 'K' $i 'lt'
done

#Tempdiff in GAIA, no correction
for i in {-50..50..10}; do
     python bash_stan.py 'astero' 10000 'None' 'GAIA' $i 'lt'
done
#
#Temp diff in GAIA, with correction
for i in {-50..50..10}; do
     python bash_stan.py 'astero' 10000 'RC' 'GAIA' $i 'lt'
done


# #Tempdiff in J, no correction
# for i in {-50..50..10}; do
#      python bash_stan.py 1000 'None' 'J' $i
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

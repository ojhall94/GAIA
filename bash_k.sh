#!/bin/bash

# Run our PyStan model on some data

# positional arguments:
#   {astero,gaia}   Choice of PyStan model.
#   iters           Number of MCMC iterations in PyStan.
#   {None,RC}       Choice of corrections to the seismic scaling relations.
#   {K,J,H,GAIA}    Choice of photometric passband.
#   tempdiff        Perturbation to the temperature values in K
#
# optional arguments:
#   -h, --help      show this help message and exit
#   -t, --testing   Turn on to output results to a test_build folder
#   -u, --update    Turn on to update the PyStan model you choose to run
#   -a, --apokasc   Turn on to run on the APOKASC subsample
#   -af, --apofull  Turn on to propagate full APOKASC data

################################
#ASTEROSEISMIC MODEL
python bash_stan.py 'astero' 10000 'None' 'K' 0.0 --update
################################ YU ET AL FULL SAMPLE PROP LOG AND TEFF
#Tempdiff in K, no correction
for i in {-50..50..10}; do
     python bash_stan.py 'astero' 10000 'None' 'K' $i
done

#Temp diff in K, with correction
for i in {-50..50..10}; do
     python bash_stan.py 'astero' 10000 'RC' 'K' $i
done
################################ APOKASC TEFF ONLY
#Tempdiff in K, no correction
for i in {-50..50..10}; do
     python bash_stan.py 'astero' 10000 'None' 'K' $i -a
done

#Temp diff in K, with correction
for i in {-50..50..10}; do
     python bash_stan.py 'astero' 10000 'RC' 'K' $i -a
done

################################
#GAIA MODEL
python bash_stan.py 'gaia' 10 'None' 'K' 0.0 --update
################################ YU ET AL SAMPLE
#Tempdiff in K [Always RC corrected]
for i in {-50..50..50}; do
     python bash_stan.py 'gaia'5000 'RC' 'K' $i
done
################################ APOKASC TEFF ONLY
# Tempdiff in K [Always RC corrected]
for i in {-50..50..50}; do
     python bash_stan.py 'gaia' 5000 'RC' 'K' $i -a
done

echo 'Complete!'

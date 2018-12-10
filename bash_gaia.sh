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
# #ASTEROSEISMIC MODEL
# python bash_stan.py 'astero' 10000 'None' 'GAIA' 0.0 --update
# ################################ YU ET AL FULL SAMPLE PROP LOG AND TEFF
#Tempdiff in G, no correction
for i in {-100..100..20}; do
     python bash_stan.py 'astero' 5000 'None' 'GAIA' $i
done

#Temp diff in G, with correction
for i in {-100..100..20}; do
     python bash_stan.py 'astero' 5000 'RC' 'GAIA' $i
done
################################ APOKASC TEFF ONLY
#Tempdiff in G, no correction
for i in {-100..100..20}; do
    python bash_stan.py 'astero' 5000 'None' 'GAIA' $i -a
done

#Temp diff in G, with correction
for i in {-100..100..20}; do
    python bash_stan.py 'astero' 5000 'RC' 'GAIA' $i -a
done

###############################
#GAIA MODEL
#python bash_stan.py 'gaia' 5000 'None' 'GAIA' 0.0 --update
################################ YU ET AL SAMPLE
#Tempdiff in G [Always RC corrected]
for i in {-100..100..100}; do
     python bash_stan.py 'gaia' 5000 'RC' 'GAIA' $i
done
################################ APOKASC TEFF ONLY
# Tempdiff in G [Always RC corrected]
for i in {-100..100..100}; do
     python bash_stan.py 'gaia' 5000 'RC' 'GAIA' $i -a
done

echo 'Complete!'

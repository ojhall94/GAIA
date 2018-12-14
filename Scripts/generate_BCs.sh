#!/bin/bash
# Recalculate Bolometric Corrections for a given temperature perturbation
#
# positional arguments:
#   tempdiff             Perturbation to the temperature values in K
#   {load,unload}        Load prepares the data for BCcodes. Unload saves it to
#                        a location of choice.
#
# optional arguments:
#   -h, --help           show this help message and exit
#   -pl, --perturb_logg  If true, perturb our value of logg using seismic
#                        scaling relations for the perturbed Teff
#   -r, --reddening      If true, include reddening in the interpolation.
#                        WARNING: This is *not* required for the Hall+18 work.
#   -a, --apokasc        If true, return a set of BCs calculated for the APOKASC
#                        subsample

#Generate iteratively for a range of temperature offsets
for i in {-100..100..20}; do
    python generate_BCs.py $i 'load' -pl
    cd ~/PhD/Hacks_and_Mocks/bolometric-corrections/BCcodes/
    ./bcall
    cd ~/PhD/Gaia_Project/GAIA/Scripts/
    python generate_BCs.py $i 'unload'
done

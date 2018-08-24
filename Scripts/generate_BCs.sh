#!/bin/bash
'''
Recalculate Bolometric Corrections for a given temperature perturbation

positional arguments:
  tempdiff             Perturbation to the temperature values in K
  {load,unload}        Load prepares the data for BCcodes. Unload saves it to
                       a location of choice.

optional arguments:
  -h, --help           show this help message and exit
  -pl, --perturb_logg  If true, perturb our value of logg using seismic
                       scaling relations for the perturbed Teff
'''

#Generate iteratively for a range of temperature offsets

for i in {-50..50..10}; do
    python generate_BCs.py $i 'load' -pl
    sh ./'''Something?'''
    python generate_BCs.py $i 'unload'
done

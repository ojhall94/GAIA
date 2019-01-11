#!/bin/bash
# Recalculate Bolometric Corrections for a given temperature perturbation

#Check our seismic runs for a couple of fixed conditions

#Bolometric corrections for -110 to -60 in APOKASC
# for i in {-110..-60..10}; do
#   cd ~/PhD/Gaia_Project/GAIA/Scripts/
#   python generate_BCs.py $i 'load' -pl -a
#   cd ~/PhD/Hacks_and_Mocks/bolometric-corrections/BCcodes/
#   ./bcall
#   cd ~/PhD/Gaia_Project/GAIA/Scripts/
#   python generate_BCs.py $i 'unload'
# done

# python bash_stan.py 'astero' 5000 'RC' 'K' -110 -a
# # python bash_stan.py 'astero' 5000 'RC' 'K' -100 -a
# python bash_stan.py 'astero' 5000 'RC' 'K' -90 -a
# #
# python bash_stan.py 'astero' 5000 'RC' 'K' -80 -a
python bash_stan.py 'astero' 5000 'RC' 'K' -70 -a
# python bash_stan.py 'astero' 5000 'RC' 'K' -60 -a

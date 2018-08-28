import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Run our PyStan model on some data')
parser.add_argument('type', type=str, choices=['astero', 'gaia'],
                    help='Choice of PyStan model.')
parser.add_argument('iters', type=int,
                    help='Number of MCMC iterations in PyStan.')
parser.add_argument('corrections', type=str, choices=['None', 'RC'],
                    help='Choice of corrections to the seismic scaling relations.')
parser.add_argument('band', type=str, choices=['K','J','H','GAIA'],
                    help='Choice of photometric passband.')
parser.add_argument('tempdiff', type=float,
                    help='Perturbation to the temperature values in K')

parser.add_argument('--testing', '-t', action='store_const', const=True, default=False,
                    help='Turn on to output results to a test_build folder')
parser.add_argument('--update', '-u', action='store_const', const=True, default=False,
                    help='Turn on to update the PyStan model you choose to run')

args = parser.parse_args()

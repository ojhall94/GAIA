import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import barbershop
import glob
import os
import sys

os.environ["GAIA_TOOLS_DATA"] = '/home/oliver/PhD/Gaia_Project/data/gaia_tools'
import gaia_tools.load as gload

if __name__ == "__main__":
    print(os.getenv("GAIA_TOOLS_DATA"))
    tgas_cat = gload.tgas()

    table = rc[1].data
    for idx, im in enumerate(rc[1].data):

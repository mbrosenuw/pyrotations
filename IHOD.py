import pyrotations
import numpy as np
from scipy.sparse import issparse
import pandas as pd

# all the constants you would want to use
mu = [0.3,1,0]  # b axis
jmin = 0
jmax = 40
T = 77
lims = [0, 50]
width = 0.003
shift = 3687.037+0.0483
consts = [17.177, 0.085704, 0.0827841]
uconsts = [16.517, 0.0881541, 0.0841155]
print(uconsts)

# make a rotation object. automatically will calculate and diagonalize the hamiltonian
DMS = pyrotations.Model(consts, uconsts, mu, jmin, jmax, T, lims, width, shift)
DMS.newcalcspectrum()
DMS.plot()


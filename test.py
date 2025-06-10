import pyrotations
import numpy as np

mu = [0,1,0]
jmin = 18
jmax = 19
T = 77
lims = [-30, 30]
shift = 0
width = 0.0016
Ax = 0.186449993466079
Bz = 0.256805714117702
Cy = 0.678719102454549
consts = np.array([Ax, Bz, Cy])
uconsts = consts

a = pyrotations.Model(consts, uconsts, mu, jmin, jmax, T, lims, width, shift)
a.newcalcspectrum()
a.plot()





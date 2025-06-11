import pyrotations
import numpy as np
from scipy.sparse import issparse
import pandas as pd

# all the constants you would want to use
mu = [0, 1, 0]  # b axis
jmin = 18
jmax = 19
T = 77
lims = [-40, 40]
shift = 0
width = 0.0016
Ax = 0.186449993466079
Bz = 0.256805714117702
Cy = 0.678719102454549
consts = np.array([Ax, Bz, Cy])
uconsts = consts
# uconsts = consts*0.95 #pretend vibrational coupling
print(uconsts)

# make a rotation object. automatically will calculate and diagonalize the hamiltonian
DMS = pyrotations.Model(consts, uconsts, mu, jmin, jmax, T, lims, width, shift)

# this is how you can extract matrices (WARNING! USE .toarray() TO MAKE THEM NUMPY)!
dipole = DMS.ops.dipole
ja = DMS.ops.lsubham.ja
jb = DMS.ops.lsubham.jb
jc = DMS.ops.lsubham.jc
Hrot = DMS.ops.lsubham.Htot

# to extract the eigenbasis labels
rotbasis = DMS.ops.lsubham.diagbasis


# to save them for viewing (as a csv)
def sparse_to_csv(matrix, basis_labels, filename, threshold=1e-10):
    """Convert a sparse matrix to a labeled CSV file, dropping small elements."""
    if issparse(matrix):
        matrix = matrix.tocsr()
        matrix.data[np.abs(matrix.data) < threshold] = 0.0
        matrix.eliminate_zeros()
        matrix = matrix.todense()
    else:
        matrix[np.abs(matrix) < threshold] = 0.0
    df = pd.DataFrame(matrix, index=basis_labels, columns=basis_labels)
    df.to_csv(filename)


label_tuples = lambda basis: [str(t) for t in basis]
sparse_to_csv(ja, label_tuples(rotbasis), "Ja.csv")
sparse_to_csv(jb, label_tuples(rotbasis), "Jb.csv")
sparse_to_csv(jc, label_tuples(rotbasis), "Jc.csv")
sparse_to_csv(Hrot, label_tuples(rotbasis), "Hrot.csv")
sparse_to_csv(dipole, label_tuples(rotbasis), "dipole.csv")

# or as a numpy file
np.savez('operators', ja=ja.toarray(), jb=jb.toarray(), jc=jc.toarray(),
         Hrot=Hrot.toarray(), dipole=dipole.toarray())

# calculate the spectrum, and save to a file !
DMS.newcalcspectrum(save=True, name='DMS_J18-19')
#matplotlib plot
DMS.plot()
#interactive plot
DMS.fancyplot()

#now with differing rotational constants
consts = np.array([Ax, Bz, Cy])
uconsts = consts*0.95 #pretend vibrational coupling
DMS2 = pyrotations.Model(consts, uconsts, mu, jmin, jmax, T, lims, width, shift)
DMS2.newcalcspectrum(save=True, name='DMS_J18-19')
DMS2.plot()
DMS2.fancyplot()
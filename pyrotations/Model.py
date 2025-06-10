from .rotham import rotation as ham
from .boltzmann import getdenom, boltzmann
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from .timing import timing
from pyrotations import plib


class Model():
    def __init__(self, lconsts, uconsts, mu, jmin, jmax, T, lims, width, shift,
                 stats=[1, 1, 1, 1]):
        for name, value in locals().items():
            if name != "self":
                setattr(self, name, value)
        self.__hams()

    def __hams(self):
        self.ops = ham(self.lconsts,self.uconsts,self.jmin, self.jmax, self.mu)

    def newcalcspectrum(self, save=False, name='spectrum'):
        with timing("Spectrum Calculation") as t:
            uidxs, lidxs = self.ops.dipole.nonzero()
            couplings = self.ops.dipole.data
            denom = getdenom(self.ops.lsubham.energies, self.T)
            ulabels = [self.ops.usubham.diagbasis[i] for i in uidxs]
            llabels = [self.ops.lsubham.diagbasis[i] for i in lidxs]
            freqs = np.array([self.ops.usubham.energies[i] - self.ops.lsubham.energies[j] for i, j in zip(uidxs, lidxs)])
            boltzs = np.array([boltzmann(self.ops.lsubham.energies[j], self.T, denom) for j in lidxs])
            intensities = np.abs(couplings) ** 2 * boltzs
            spectrum_data = []
            for freq, inten, lower, upper in zip(freqs, intensities, llabels, ulabels):
                spectrum_data.append({
                    "frequency": freq,
                    "intensity": inten,
                    "lower_state": tuple(lower),
                    "upper_state": tuple(upper)
                })
            # Create DataFrame from the list of dictionaries
            # Convert accumulated list to a DataFrame at the end
            self.spectrum = pd.DataFrame(spectrum_data)
            self.spectrum["frequency"] = self.spectrum["frequency"].astype("float")
            self.spectrum["intensity"] = self.spectrum["intensity"].astype("float")
            self.spectrum = self.spectrum.sort_values(by="frequency", ascending=True)
            self.spectrum = self.spectrum[(self.spectrum['intensity'] > self.spectrum['intensity'].max() * 10**(-3))]
            self.spectrum = self.spectrum[
                (self.spectrum['frequency'] >= self.lims[0]) & (self.spectrum['frequency'] <= self.lims[1])]
            print(len(self.spectrum), ' transitions evaluated.')

            # Save the spectrum to CSV
            if save:
                self.spectrum.to_csv(name + '.csv', index=False)
            return self.spectrum


    def fancyplot(self, other=None):
        plib.fplotter(self.spectrum, self.lims, self.width, self.shift, 'Dimethyl Sulfide Spectrum', other)

    def plot(self, other = None, title = 'Dimethyl Sulfide Spectrum (with Torsions)'):
        spec, x = plib.stdplotter(self.spectrum, self.lims, self.width, self.shift, title, other)
        return spec, x



def l0(basis_labels):
    n = len(basis_labels)
    l0_indices = [i for i, label in enumerate(basis_labels) if label[1] == 0]
    m = len(l0_indices)
    data = np.ones(m)
    row_indices = l0_indices
    col_indices = list(range(m))
    P = csr_matrix((data, (row_indices, col_indices)), shape=(n, m))
    l0_basis_labels = [basis_labels[i] for i in l0_indices]
    return P, l0_basis_labels

def op(lPT, uPT, operator, km=None, epsilon=1):
    if km is None:
        km = lPT.kmax
    else:
        km = min(km, lPT.kmax)

    lWs = [epsilon ** j * lPT.W_list[j] for j in range(km + 1)]
    uWs = [epsilon ** j * uPT.W_list[j] for j in range(km + 1)]

    O = sum(
        uWs[l].conj().T @ operator @ lWs[k - l]
        for k in range(km + 1)
        for l in range(k + 1)
    )
    return O

def getfullblocks(basis):
    blocks = {}
    for idx, (m, l, gamma1, p, j, gamma2, gamma0) in enumerate(basis):
        if (l, j, gamma0) not in blocks:
            blocks[(l, j, gamma0)] = {"start": idx, "end": idx, "count": 1}
        else:
            blocks[(l, j, gamma0)]["end"] = idx
            blocks[(l, j, gamma0)]["count"] += 1
    return blocks

from .rotham import rotation as ham
from .boltzmann import getdenom, boltzmann
import numpy as np
import pandas as pd
from .timing import timing
from . import plib


class Model():
    """
    Rotational spectroscopy model for simulating rovibrational transitions using asymmetric top Hamiltonians.

    Parameters
    ----------
    lconsts : np.array of float
        Rotational constants (A'', B'', C'') for the lower vibrational state [cm^-1].
    uconsts : np.array of float
        Rotational constants (A', B', C') for the upper vibrational state [cm^-1].
    mu : np.array of float
        Dipole moment components along the a-, b-, and c-principal axes [rel].
    jmin : int
        Minimum total angular momentum quantum number J to include in the calculation.
    jmax : int
        Maximum total angular momentum quantum number J to include in the calculation.
    T : float
        Temperature in Kelvin for computing Boltzmann populations.
    lims : np.array of float
        Frequency axis limits for plotting the computed spectrum [cm^-1].
    width : float
        Full width at half maximum (FWHM) for the spectral lineshape [cm^-1].
    shift : float
        Band origin or vibrational transition energy [cm^-1].
    stats : list of int, optional
        Nuclear spin statistical weights for each irreducible representation (e.g., [ee,eo,oe,oo]).
        Default is [1, 1, 1, 1].

    Attributes
    ----------
    ops : rotation
        Rotation object containing all state-resolved Hamiltonian and dipole data.

        ops.lsubham : subham
            Hamiltonian and operator data for the lower vibrational state.
            - wfns : csr_matrix
                Eigenfunctions matrix for the lower Hamiltonian.
            - energies : np.ndarray
                Eigenenergies for each lower state.
            - Htot : csr_matrix
                Diagonal matrix of eigenenergies.
            - ja, jb, jc : csr_matrix
                Angular momentum operator matrices in the eigenbasis.
            - I : csr_matrix
                Identity operator matrix.
            - diagbasis : list[tuple[int, int, str]]
                Indexed quantum numbers (i, J, Γ) of diagonal states.
            - basis : list[tuple[int, int, str]]
                Symmetry-adapted angular momentum basis (J, K, Γ).

        ops.usubham : subham
            Same structure as `lsubham`, but for the upper vibrational state.

        ops.dipole : csr_matrix
            Matrix representation of transition dipole operator between lower and upper states.

    spectrum : pd.DataFrame
        DataFrame storing transition information (created by `newcalcspectrum()`):
            - frequency : float
                Transition frequency [cm^{-1}].
            - intensity : float
                Transition intensity based on dipole moment and Boltzmann factor.
            - lower_state : tuple[int, int, str]
                Quantum labels of the initial (lower) state (idx, J, \Gamma).
            - upper_state : tuple[int, int, str]
                Quantum labels of the final (upper) state (idx, J, \Gamma).
    """
    def __init__(self, lconsts, uconsts, mu, jmin, jmax, T, lims, width, shift,
                 stats=[1, 1, 1, 1]):
        """
        Initialize a rotational spectroscopy model for a rovibrational transition.

        Parameters
        ----------
        lconsts : np.array of float
            Rotational constants (A'', B'', C'') for the lower vibrational state [cm^-1].
        uconsts : np.array of float
            Rotational constants (A', B', C') for the upper vibrational state [cm^-1].
        mu : np.array of float
            Dipole moment components along the a-, b-, and c-principal axes [rel].
        jmin : int
            Minimum total angular momentum quantum number J to include in the calculation.
        jmax : int
            Maximum total angular momentum quantum number J to include in the calculation.
        T : float
            Temperature in Kelvin for computing Boltzmann populations.
        lims : np.array of float
            Frequency axis limits for plotting the computed spectrum [cm^-1].
        width : float
            Full width at half maximum (FWHM) for the spectral lineshape [cm^-1].
        shift : float
            Band origin or vibrational transition energy [cm^-1].
        stats : list of int, optional
            Nuclear spin statistical weights for each irreducible representation (e.g., [ee,eo,oe,oo]).
            Default is [1, 1, 1, 1].
        """
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

            self.spectrum = pd.DataFrame(spectrum_data)
            self.spectrum["frequency"] = self.spectrum["frequency"].astype("float")
            self.spectrum["intensity"] = self.spectrum["intensity"].astype("float")
            self.spectrum = self.spectrum.sort_values(by="frequency", ascending=True)
            self.spectrum = self.spectrum[(self.spectrum['intensity'] > self.spectrum['intensity'].max() * 10**(-6))]
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


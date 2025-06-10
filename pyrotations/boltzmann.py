import numpy as np
from scipy.constants import physical_constants as pc

Kb = pc['Boltzmann constant in inverse meter per kelvin'][0]/100

def getdenom(energies,T):
    def calcelem(energy):
        return np.exp(-energy / (Kb * T))
    vc = np.vectorize(calcelem)
    denom = np.sum(vc(energies))
    return denom

def boltzmann(E, T, denom):
    num = np.exp(-E/(Kb*T))
    return num/denom

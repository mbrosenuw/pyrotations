import numpy as np
from . import rotsymmetrize as sym
from scipy.sparse import csr_matrix
from scipy.linalg import block_diag
from .dipole import getdipole as getdipole
import scipy.sparse as sp


def jsquare(j):
    mat = np.zeros((2*j+1,2*j+1))
    for a in range(mat.shape[0]):
        for b in range(mat.shape[1]):
            if a == b:
                mat[a,b] = j * (j + 1)
    return mat

def jb(j):
    kvec = np.arange(-j, j + 1, 1)
    mat = np.zeros((2*j+1,2*j+1))
    for i in range(mat.shape[0]):
        mat[i,i] = kvec[i]
    return mat

def jbodyplus(j, kvec):
    mat = np.zeros((2*j+1,2*j+1))
    for a in range(mat.shape[0]):
        for b in range(mat.shape[1]):
            if a == b-1:
                mat[a,b] = (j * (j+1) - kvec[b]*(kvec[b]-1))**(1/2)
    return mat

def jbodyminus(j, kvec):
    mat = np.zeros((2*j+1,2*j+1))
    for a in range(mat.shape[0]):
        for b in range(mat.shape[1]):
            if a == b+1:
                mat[a,b] = (j * (j+1) - kvec[b]*(kvec[b]+1))**(1/2)
    return mat

def ja(j):
    kvec = np.arange(-j, j + 1, 1)
    mat = 1j/2 *(jbodyplus(j,kvec) - jbodyminus(j,kvec))
    return mat

def jc(j):
    kvec = np.arange(-j, j + 1, 1)
    mat = 1 / 2 * (jbodyplus(j, kvec) + jbodyminus(j, kvec))
    return mat

def getH(consts, j):
    H = (consts[1] * np.linalg.matrix_power(jb(j),2) + consts[0] *np.linalg.matrix_power(ja(j),2)
          + consts[2] *np.linalg.matrix_power(jc(j),2))
    return H

class subrotation:
    """
    Attributes
    ----------
    consts : tuple[float, float, float]
        Rotational constants (A, B, C) for this vibrational state [cm^-1].

    j : int
        Total angular momentum quantum number J associated with this block.

    ja : np.ndarray
        Operator matrix for J_a in the eigenbasis (after symmetry adaptation and diagonalization).

    jb : np.ndarray
        Operator matrix for J_b in the eigenbasis.

    jc : np.ndarray
        Operator matrix for J_c in the eigenbasis.

    I : np.ndarray
        Identity matrix in the symmetrized basis for use in operator construction.

    Hrot : np.ndarray
        Full rotational Hamiltonian in the symmetry-adapted basis.

    wfns : np.ndarray
        Matrix whose columns are the eigenvectors of Hrot, block-diagonalized by (J, \Gamma).

    energies : np.ndarray
        Rotational energy eigenvalues corresponding to each eigenfunction in `wfns`.

    basis : list[tuple[int, int, int]]
        List of tuples (J, K, \Gamma) labeling the original basis states.

    diagbasis : list[tuple[int, int, int]]
        List of tuples (i, J, \Gamma) labeling each eigenstate by index and its symmetry.
    """
    def __init__(self, consts, j):
        self.consts = consts
        self.j = j
        S, symbasis = sym.gettrans(j)

        self.ja = S @ ja(j) @ S.T
        self.jb = S @ jb(j) @ S.T
        self.jc = S @ jc(j) @ S.T
        self.I = np.eye(len(symbasis), len(symbasis))
        self.Hrot = (consts[1] * np.linalg.matrix_power(self.jb,2) + consts[0] *np.linalg.matrix_power(self.ja,2)
          + consts[2] *np.linalg.matrix_power(self.jc,2))
        blocks = getrotblocks(symbasis)

        self.wfns = np.zeros(self.Hrot.shape, dtype=np.complex128)
        self.energies = np.zeros(self.wfns.shape[0])
        self.diagbasis = [None] * len(symbasis)
        for (j,gamma), block in blocks.items():
            idxs = block['start']
            idxe = block['end'] + 1
            self.energies[idxs:idxe], self.wfns[idxs:idxe, idxs:idxe] = np.linalg.eigh(self.Hrot[idxs:idxe, idxs:idxe])
            self.diagbasis[idxs:idxe] = [(i,j,gamma) for i in range(idxe - idxs)]
        self.basis = symbasis
        self.ja = self.wfns.conj().T @ self.ja @ self.wfns
        self.jb = self.wfns.conj().T @ self.jb @ self.wfns
        self.jc = self.wfns.conj().T @ self.jc @ self.wfns


class rotation:
    """
    Assembles full lower and upper vibrational-state Hamiltonians over a range of J values,
    and constructs the transition dipole matrix in the eigenbasis.

    Attributes
    ----------
    lsubham : subham
        Complete rotational Hamiltonian and operator set for the lower vibrational level.

    usubham : subham
        Complete rotational Hamiltonian and operator set for the upper vibrational level.

    dipole : scipy.sparse.csr_matrix
        Dipole transition operator in the eigenbasis, constructed from body-frame components
        and transformed via the eigenvectors of the lower and upper states.
    """
    def __init__(self, lconsts, uconsts, jmin, jmax, mu):

        lsubrots = [subrotation(lconsts, j) for j in range(jmin, jmax+1)]
        self.lsubham = getfullmatrices(lsubrots)
        usubrots = [subrotation(uconsts, j) for j in range(jmin, jmax + 1)]
        self.usubham = getfullmatrices(usubrots)
        self.dipole, dipolebasis = getdipole(jmin, jmax, mu)
        self.dipole = (self.usubham.wfns.conj().T @ self.dipole @ self.lsubham.wfns).tocsr()



def getrotblocks(basis):
    blocks = {}
    for idx, (j,k, gamma) in enumerate(basis):
        if (j,gamma) not in blocks:
            blocks[(j,gamma)] = {"start": idx, "end": idx, "count": 1}
        else:
            blocks[(j,gamma)]["end"] = idx
            blocks[(j,gamma)]["count"] += 1
    return blocks

def getfullmatrices(syslist):
    wfnslist = [sys.wfns for sys in syslist]
    wfns = csr_matrix(block_diag(*wfnslist))
    elist = [sys.energies for sys in syslist]
    energies = np.hstack(elist)
    Htot = sp.diags(energies, offsets=0, format='csr')
    basislist = [item for sys in syslist for item in sys.basis]
    diagbasislist = [item for sys in syslist for item in sys.diagbasis]
    jalist = [sys.ja for sys in syslist]
    ja = csr_matrix(block_diag(*jalist))
    jblist = [sys.jb for sys in syslist]
    jb = csr_matrix(block_diag(*jblist))
    jclist = [sys.jc for sys in syslist]
    jc = csr_matrix(block_diag(*jclist))
    Ilist = [sys.I for sys in syslist]
    I = csr_matrix(block_diag(*Ilist))
    return subham(wfns, energies,Htot, ja, jb, jc, I, diagbasislist, basislist)

class subham:
    """
    Container for the full rotational Hamiltonian, eigenvectors, and operators
    across all J values.

    Attributes
    ----------
    wfns : scipy.sparse.csr_matrix
        Block-diagonal matrix of eigenvectors for each J block.

    energies : np.ndarray
        Energy eigenvalues corresponding to each eigenfunction in the basis.

    Htot : scipy.sparse.csr_matrix
        Diagonal Hamiltonian matrix with energy eigenvalues on the diagonal.

    ja : scipy.sparse.csr_matrix
        Operator matrix for J_a.

    jb : scipy.sparse.csr_matrix
        Operator matrix for J_b.

    jc : scipy.sparse.csr_matrix
        Operator matrix for J_c.

    I : scipy.sparse.csr_matrix
        Block-diagonal identity matrix matching the operator dimensions.

    diagbasis : list[tuple[int, int, int]]
        List of (index, J, \Gamma) labels for the eigenstates.

    basis : list[tuple[int, int, int]]
        List of (J, K, \Gamma) labels for the original symmetric top basis states.
    """
    def __init__(self,wfns, energies,Htot, ja, jb, jc, I, diagbasis, basis):
        for name, value in locals().items():
            if name != "self":
                setattr(self, name, value)

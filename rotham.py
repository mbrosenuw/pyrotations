import numpy as np
from Code.Hamiltonians.symmetrizers import rotsymmetrize as sym
from scipy.sparse import csr_matrix
from scipy.linalg import block_diag
from Code.Spectrum_Generation.Line_Strengths.dipole import getdipole as getdipole
import scipy.sparse as sp


def jsquare(j):
    mat = np.zeros((2*j+1,2*j+1))
    for a in range(mat.shape[0]):
        for b in range(mat.shape[1]):
            if a == b:
                mat[a,b] = j * (j + 1)
    # mat, _ = sym.Htransform(mat)
    # T, _ = sym.gettrans(j)
    # mat = T @ mat @ T.T
    return mat

def jb(j):
    kvec = np.arange(-j, j + 1, 1)
    mat = np.zeros((2*j+1,2*j+1))
    for i in range(mat.shape[0]):
        mat[i,i] = kvec[i]
    # mat, _ = sym.Htransform(mat)
    # T, _ = sym.gettrans(j)
    # mat = T @ mat @ T.T
    return mat

def jbodyplus(j, kvec):
    mat = np.zeros((2*j+1,2*j+1))
    for a in range(mat.shape[0]):
        for b in range(mat.shape[1]):
            if a == b-1:
                mat[a,b] = (j * (j+1) - kvec[b]*(kvec[b]-1))**(1/2)

    # mat, _ = sym.Htransform(mat)
    # T, _ = sym.gettrans(j)
    # mat = T @ mat @ T.T
    return mat

def jbodyminus(j, kvec):
    mat = np.zeros((2*j+1,2*j+1))
    for a in range(mat.shape[0]):
        for b in range(mat.shape[1]):
            if a == b+1:
                mat[a,b] = (j * (j+1) - kvec[b]*(kvec[b]+1))**(1/2)
    # mat, _ = sym.Htransform(mat)
    # T, _ = sym.gettrans(j)
    # mat = T @ mat @ T.T
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
    def __init__(self, lconsts, uconsts, jmin, jmax, mu):

        lsubrots = [subrotation(lconsts, j) for j in range(jmin, jmax+1)]
        self.lwfns, self.lHtot, self.lja, self.ljb, self.ljc, self.lI, self.ldiagbasis, self.lbasis = getfullmatrices(lsubrots)
        usubrots = [subrotation(uconsts, j) for j in range(jmin, jmax + 1)]
        self.uwfns, self.uHtot, self.uja, self.ujb, self.ujc, self.uI, self.udiagbasis, self.ubasis = getfullmatrices(
            usubrots)
        self.dipole, dipolebasis = getdipole(jmin, jmax, mu)
        self.dipole = self.uwfns.conj().T @ self.dipole @ self.lwfns



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
    return wfns, Htot, ja, jb, jc, I, diagbasislist, basislist


# rotation([1,2,3],[1,2,3], 1,10, [0,1,0])
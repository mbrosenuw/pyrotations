import numpy as np
from sympy.physics.wigner import wigner_3j as tj
import time
from scipy.linalg import block_diag
import os
from scipy.sparse import csr_matrix
from . import rotsymmetrize as sym
from importlib.util import find_spec
import importlib.resources as res  # requires Python >=3.7


def tjk(basis, qk):
    tic = time.time()
    mat = np.zeros((len(basis), len(basis)), dtype=complex)
    jrange = list(set(j for j, k in basis))
    basis_dict = {pair: i for i, pair in enumerate(basis)}
    for (jp, kp) in basis:
        jpprange = [jpp for jpp in jrange if np.abs(jp - 1) <= jpp <= jp + 1]
        upperbasis = [(jpp, kpp) for jpp in jpprange for kpp in range(-jpp, jpp + 1) if (-kp + qk + kpp) == 0]

        for (jpp, kpp) in upperbasis:
            if (jpp, kpp) in basis_dict:
                idxpp = basis_dict[(jpp, kpp)]
                idxp = basis_dict[(jp, kp)]
                mat[idxp, idxpp] = tj(jp, 1, jpp, -kp, qk, kpp).evalf(50)
    toc = time.time()
    elapsed_time = toc - tic
    print(elapsed_time, 'seconds elapsed for smart')
    return mat


def redmat(basis):
    tic = time.time()
    basis_dict = {pair: i for i, pair in enumerate(basis)}
    jrange = list(set(j for j, k in basis))
    mat = np.zeros((len(basis), len(basis)), dtype=complex)
    for jp in jrange:
        for jpp in jrange:
            factor = np.sqrt(2 * jpp + 1) * np.sqrt(2 * jp + 1)
            indices_p = [basis_dict[(jp, kp)] for kp in set(k for j, k in basis if j == jp)]
            indices_pp = [basis_dict[(jpp, kpp)] for kpp in set(k for j, k in basis if j == jpp)]
            for idxp in indices_p:
                for idxpp in indices_pp:
                    mat[idxp, idxpp] = factor

    toc = time.time()
    elapsed_time = toc - tic
    print(f"Reduced J matrix generation time: {elapsed_time:.4f} seconds")
    return mat


def kphase(basis):
    tic = time.time()
    mat = np.zeros((len(basis), len(basis)), dtype=complex)
    basis_dict = {pair: i for i, pair in enumerate(basis)}
    for (jp, kp) in basis:
        for (jpp, kpp) in basis:
            idxpp = basis_dict[(jpp, kpp)]
            idxp = basis_dict[(jp, kp)]
            mat[idxp, idxpp] = (-1) ** (-kp)
    toc = time.time()
    elapsed_time = toc - tic
    print(f"Kphase matrix generation time: {elapsed_time:.4f} seconds")
    return mat

def get_data_path(filename: str) -> str:
    """Get the absolute path to a data file inside the pyrotations package."""
    if find_spec("pyrotations") is not None:
        try:
            # Python 3.9+ recommended
            return str(res.files("pyrotations").joinpath(filename))
        except AttributeError:
            # For Python 3.7–3.8 fallback to legacy
            with res.path("pyrotations", filename) as p:
                return str(p)
    else:
        # fallback: relative path for dev use
        return os.path.join(os.path.dirname(__file__), filename)


# Function to save data to a .npz file
def save_data(filename, basis, tjk0, tjkm1, tjkp1, rmat, kmat):
    np.savez(filename, basis=np.array(basis, dtype=object), tjk0=tjk0, tjkm1=tjkm1, tjkp1=tjkp1, rmat=rmat,
             kmat=kmat)
    print(f"Data saved to {filename}")


# Function to load data from a .npz file
def load_data(filename):
    data = np.load(filename, allow_pickle=True)
    return data['basis'], data['tjk0'], data['tjkm1'], data['tjkp1'], data['rmat'], data['kmat']


def getdipole(jmin, jmax, mu):
    jlist = [j for j in range(jmin, jmax + 1)]
    basis = [(j, k) for j in jlist for k in range(-j, j + 1)]
    filename = get_data_path("3jsyms_j30.npz")
    if os.path.exists(filename):
        loaded_basis, ltjk0, ltjkm1, ltjkp1, rmat, kmat = load_data(filename)
    else:
        jlist2 = [j for j in range(31)]
        genbasis = [(j, k) for j in jlist2 for k in range(-j, j + 1)]
        tjk0 = tjk(genbasis, 0)
        tjkm1 = tjk(genbasis, -1)
        tjkp1 = tjk(genbasis, 1)
        rmat = redmat(genbasis)
        kmat = kphase(genbasis)
        save_data(filename, genbasis, tjk0, tjkm1, tjkp1, rmat, kmat)
        loaded_basis, ltjk0, ltjkm1, ltjkp1, rmat, kmat = load_data(filename)

    loaded_basis = [tuple(pair) for pair in loaded_basis]
    mapmat = np.zeros((len(basis), len(loaded_basis)))

    # Populate the matrix with 1's where elements in userbasis are found in loaded_basis
    for i, user_elem in enumerate(basis):
        for j, loaded_elem in enumerate(loaded_basis):
            if user_elem == loaded_elem:
                mapmat[i, j] = 1

    ltjk0 = mapmat @ ltjk0 @ mapmat.T
    ltjkm1 = mapmat @ ltjkm1 @ mapmat.T
    ltjkp1 = mapmat @ ltjkp1 @ mapmat.T
    rmat = mapmat @ rmat @ mapmat.T
    kmat = mapmat @ kmat @ mapmat.T


    Ts = [a for a, _ in (sym.gettrans(j) for j in jlist)]
    symbasis = [item for _, a in (sym.gettrans(j) for j in jlist) for item in a]
    fulltrans = block_diag(*Ts)

    dipole = mu[1] * ltjk0 + mu[0] * 1 / 2 * (ltjkp1 - ltjkm1) + mu[2] * 1 / (2j) * (ltjkp1 + ltjkm1)
    dipole = kmat * rmat * dipole
    dipole = np.round(fulltrans @ dipole @ fulltrans.T,12)
    return csr_matrix(dipole), symbasis
    # return csr_matrix(dipole), basis

def getdipole2(jmin, jmax, mu):
    jlist2 = [j for j in range(jmin, jmax +1 )]
    genbasis = [(j, k) for j in jlist2 for k in range(-j, j + 1)]
    tjk0 = tjk(genbasis, 0)
    tjkm1 = tjk(genbasis, -1)
    tjkp1 = tjk(genbasis, 1)
    rmat = redmat(genbasis)
    kmat = kphase(genbasis)


    Ts = [a for a, _ in (sym.gettrans(j) for j in jlist2)]
    symbasis = [item for _, a in (sym.gettrans(j) for j in jlist2) for item in a]
    fulltrans = block_diag(*Ts)

    dipole = mu[1] * tjk0 + mu[0] * 1 / 2 * (tjkp1 - tjkm1) + mu[2] * 1 / (2j) * (tjkp1 + tjkm1)
    dipole = kmat * rmat * dipole
    dipole = np.round(fulltrans @ dipole @ fulltrans.T,12)
    return csr_matrix(dipole), symbasis
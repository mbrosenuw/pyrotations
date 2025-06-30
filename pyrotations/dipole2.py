import numpy as np
from sympy.physics.wigner import wigner_3j as tj
import time
import os
from scipy.sparse import coo_matrix, csr_matrix, block_diag, save_npz, load_npz
from pyrotations import rotsymmetrize as sym
from importlib.util import find_spec
import importlib.resources as res  # requires Python >=3.7
from .timing import timing

def tjk(basis, qk):
    tic = time.time()
    rows, cols, data = [], [], []
    jrange = list(set(j for j, k in basis))
    basis_dict = {pair: i for i, pair in enumerate(basis)}
    for (jp, kp) in basis:
        jpprange = [jpp for jpp in jrange if abs(jp - 1) <= jpp <= jp + 1]
        for jpp in jpprange:
            for kpp in range(-jpp, jpp + 1):
                if (-kp + qk + kpp) == 0:
                    if (jpp, kpp) in basis_dict:
                        idxp = basis_dict[(jp, kp)]
                        idxpp = basis_dict[(jpp, kpp)]
                        val = tj(jp, 1, jpp, -kp, qk, kpp).evalf(50)
                        rows.append(idxp)
                        cols.append(idxpp)
                        data.append(complex(val))
    mat = coo_matrix((data, (rows, cols)), shape=(len(basis), len(basis)), dtype=complex).tocsr()
    print(f"{time.time() - tic:.2f} seconds elapsed for smart")
    return mat

def redmat(basis):
    tic = time.time()
    basis_dict = {pair: i for i, pair in enumerate(basis)}
    jrange = list(set(j for j, k in basis))
    rows, cols, data = [], [], []
    for jp in jrange:
        for jpp in jrange:
            factor = np.sqrt(2 * jpp + 1) * np.sqrt(2 * jp + 1)
            for kp in set(k for j, k in basis if j == jp):
                for kpp in set(k for j, k in basis if j == jpp):
                    idxp = basis_dict[(jp, kp)]
                    idxpp = basis_dict[(jpp, kpp)]
                    rows.append(idxp)
                    cols.append(idxpp)
                    data.append(factor)
    mat = coo_matrix((data, (rows, cols)), shape=(len(basis), len(basis)), dtype=complex).tocsr()
    print(f"Reduced J matrix generation time: {time.time() - tic:.4f} seconds")
    return mat

def kphase(basis):
    tic = time.time()
    basis_dict = {pair: i for i, pair in enumerate(basis)}
    rows, cols, data = [], [], []
    for (jp, kp) in basis:
        idxp = basis_dict[(jp, kp)]
        for (jpp, kpp) in basis:
            idxpp = basis_dict[(jpp, kpp)]
            rows.append(idxp)
            cols.append(idxpp)
            data.append((-1) ** (-kp))
    mat = coo_matrix((data, (rows, cols)), shape=(len(basis), len(basis)), dtype=complex).tocsr()
    print(f"Kphase matrix generation time: {time.time() - tic:.4f} seconds")
    return mat

def get_data_path(filename: str) -> str:
    if find_spec("pyrotations") is not None:
        try:
            return str(res.files("pyrotations").joinpath(filename))
        except AttributeError:
            with res.path("pyrotations", filename) as p:
                return str(p)
    else:
        return os.path.join(os.path.dirname(__file__), filename)

def save_data(filename_prefix, basis, tjk0, tjkm1, tjkp1, rmat, kmat):
    np.save(f'{filename_prefix}_basis.npy', np.array(basis, dtype=object))
    save_npz(f'{filename_prefix}_tjk0.npz', tjk0)
    save_npz(f'{filename_prefix}_tjkm1.npz', tjkm1)
    save_npz(f'{filename_prefix}_tjkp1.npz', tjkp1)
    save_npz(f'{filename_prefix}_rmat.npz', rmat)
    save_npz(f'{filename_prefix}_kmat.npz', kmat)
    print(f"Data saved with prefix {filename_prefix}")

def load_data(filename_prefix):
    basis = np.load(f'{filename_prefix}_basis.npy', allow_pickle=True)
    tjk0 = load_npz(f'{filename_prefix}_tjk0.npz')
    tjkm1 = load_npz(f'{filename_prefix}_tjkm1.npz')
    tjkp1 = load_npz(f'{filename_prefix}_tjkp1.npz')
    rmat = load_npz(f'{filename_prefix}_rmat.npz')
    kmat = load_npz(f'{filename_prefix}_kmat.npz')
    return basis, tjk0, tjkm1, tjkp1, rmat, kmat

def getdipole(jmin, jmax, mu):
    jlist = [j for j in range(jmin, jmax + 1)]
    basis = [(j, k) for j in jlist for k in range(-j, j + 1)]
    filename_prefix = get_data_path("3jsyms_j70")

    regenerate = True
    if os.path.exists(f"{filename_prefix}_basis.npy"):
        try:
            loaded_basis, *_ = load_data(filename_prefix)
            loaded_basis = [tuple(pair) for pair in loaded_basis]
            loaded_jset = set(j for j, _ in loaded_basis)
            expected_jset = set(jlist)
            if expected_jset.issubset(loaded_jset):
                regenerate = False
        except Exception:
            # Fail safe: regenerate if loading fails
            pass

    if regenerate:
        with timing('Generating dipole matrix'):
            full_jlist = list(range(jmax + 1))
            genbasis = [(j, k) for j in full_jlist for k in range(-j, j + 1)]
            tjk0 = tjk(genbasis, 0)
            tjkm1 = tjk(genbasis, -1)
            tjkp1 = tjk(genbasis, 1)
            rmat = redmat(genbasis)
            kmat = kphase(genbasis)
            save_data(filename_prefix, genbasis, tjk0, tjkm1, tjkp1, rmat, kmat)

    loaded_basis, ltjk0, ltjkm1, ltjkp1, rmat, kmat = load_data(filename_prefix)
    loaded_basis = [tuple(pair) for pair in loaded_basis]

    # Map requested basis indices to loaded basis indices
    idx_map = {elem: i for i, elem in enumerate(loaded_basis)}
    rows, cols = zip(*[(i, idx_map[elem]) for i, elem in enumerate(basis)])
    data = np.ones(len(rows))
    mapmat = coo_matrix((data, (rows, cols)), shape=(len(basis), len(loaded_basis))).tocsr()

    # Restrict matrices to requested basis subset
    ltjk0 = mapmat @ ltjk0 @ mapmat.T
    ltjkm1 = mapmat @ ltjkm1 @ mapmat.T
    ltjkp1 = mapmat @ ltjkp1 @ mapmat.T
    rmat = mapmat @ rmat @ mapmat.T
    kmat = mapmat @ kmat @ mapmat.T

    with timing('Symmetrizing dipole matrix'):
        Ts = [a for a, _ in (sym.gettrans(j) for j in jlist)]
        symbasis = [item for _, a in (sym.gettrans(j) for j in jlist) for item in a]
        fulltrans = block_diag(Ts, format='csr')

    with timing('Constructing final dipole matrix'):
        dipole = mu[1] * ltjk0 + mu[0] * 0.5 * (ltjkp1 - ltjkm1) + mu[2] * (0.5j) * (ltjkp1 + ltjkm1)
        dipole = kmat.multiply(rmat).multiply(dipole)
        dipole = fulltrans @ dipole @ fulltrans.T
        dipole.data[np.abs(dipole.data) < 1e-12] = 0
        dipole.eliminate_zeros()

    return dipole, symbasis


import numpy as np

def Htransform(mat):
    n = mat.shape[0]
    j = (n-1)/2
    u = np.zeros((n, n), dtype=complex)
    if j %2 == 0:
        u[n // 2, n // 2] = 1
    else:
        u[n // 2, n // 2] = 1j
    for k in range(n // 2):
        u[k, k] = (-1)**j *np.sqrt(2) / 2
        u[k, n - k - 1] = np.sqrt(2) / 2
        u[n - k - 1, k] = (-1)**j *-1j * np.sqrt(2) / 2
        u[n - k - 1, n - k - 1] = 1j * np.sqrt(2) / 2
    kidxs = np.arange(0, n, 1)
    evenkidx = np.arange(j%2, n,2)
    oddkidx = kidxs[~np.isin(kidxs, evenkidx)]
    mp = len(oddkidx)//2
    newidxs = np.concatenate((oddkidx[:mp], evenkidx, oddkidx[mp:])).astype('int')
    A = np.matrix(np.zeros((n,n)))
    ks = genks(int(j))
    ks2 = np.zeros((int(2*j+1),2))
    for i in range(n):
        A[i,newidxs[i]] = 1
        ks2[i] = ks[newidxs[i]][:]
    u = A @ u
    return u.H @ mat @ u, ks2

def genks(j):
    ks = np.zeros((2*j+1, 2))
    if j %2 ==0:
        ks[j] = (0,0)
    else:
        ks[j] = (0,1)
    for i in range(1,j+1):
        ks[j-i] = (i,0)
        ks[j + i] = (i,1)
    return ks

def genks2(j):
    kvec = np.arange(0,j+1)
    svec = np.arange(0,2)
    basis = [(j,k,s) for k in kvec for s in svec]
    return basis


def generate_primitive_basis(j):
    """Generate the list of primitive basis states (n1, n2)."""
    states = [(j,k) for k in range(-j, j + 1)]
    return states


def symmetrized_basis(j, k, gamma):
    """Return the symmetrized basis state as a vector in the primitive basis."""
    exist = True
    if k == 0:
        if j %2 == 0:
            if gamma == 0:
                coeff = [1]
            else:
                coeff = [0]
                exist = False
            states = [
                (j,k)
            ]
        else:
            if gamma == 1:
                coeff = [1]
            else:
                coeff = [0]
                exist = False
            states = [
                (j, k)
            ]
    else:
        if k %2 ==0:
            if gamma == 0:
                coeff = [np.sqrt(1/2)*1, np.sqrt(1/2)*(-1) ** j]
            elif gamma == 1:
                coeff = [np.sqrt(1/2) * 1, np.sqrt(1/2) * (-1) ** (j + 1)]
            else:
                coeff = [0,0]
                exist = False
        else:
            if gamma == 2:
                coeff = [np.sqrt(1/2)*1, np.sqrt(1/2)*(-1) ** (j+1)]
            elif gamma == 3:
                coeff = [np.sqrt(1/2) * 1, np.sqrt(1/2) * (-1) ** (j)]
            else:
                coeff = [0, 0]
                exist = False
        states = [
            (j, k),
            (j, -k)
        ]

    return coeff, states, exist


def gettrans(j):
    """Build the transformation matrix from primitive to symmetrized basis."""
    primitive_basis = generate_primitive_basis(j)
    # for b in primitive_basis: print(b)
    symmetrized_basis_list = []
    T = []
    for gamma in [0, 1, 2, 3]:
        for k in range(0, j + 1):
            coeff, states, exist = symmetrized_basis(j,k, gamma)
            if exist == True:
                row = np.zeros(len(primitive_basis))
                for c, (p1, p2) in zip(coeff, states):
                    if (p1, p2) in primitive_basis:
                        idx = primitive_basis.index((p1, p2))
                        row[idx] += c
                T.append(row)
                symmetrized_basis_list.append((j,k,gamma))
    return np.array(T), symmetrized_basis_list
import numpy as np
import scipy
# from cvxopt import matrix, solvers
from tqdm import tqdm

def unmix_LS_unconstrained(M, abs):
    '''
    Unmixing algorithm using least squares with no constraints.
    input:
        M: endmember spectra matrix, np.array of shape (k, n) where k is the number of spectral bands and n is the number of endmembers
        abs: absorbance spectra, np.array of shape (...,k) where ... is the number of pixels
    output:
        c: estimated abundances, np.array of shape (...,n) where ... is the number of pixels
        err: difference spectrum, np.array of shape (...,k) where ... is the number of pixels
    '''
    M_inv = np.linalg.pinv(M)
    c = np.einsum("ij,...j->...i", M_inv, abs)
    err = np.einsum("kn,...n->...k", M, c) - abs
    return c, err

def unmix_LS_nonnegative(M, abs):
    '''
    Unmixing algorithm using least squares with non-negative constraints.
    input:
        M: endmember spectra matrix, np.array of shape (k, n) where k is the number of spectral bands and n is the number of endmembers
        abs: absorbance spectra, np.array of shape (k) or (m,l,k) where m and l are the number of pixels
    output:
        c: estimated abundances, np.array of shape (n) or (m,l,n)
        err: difference spectrum, np.array of shape (k) or (m,l,k)
    '''
    n = M.shape[1]
    if abs.ndim == 1:
        c, _ = scipy.optimize.nnls(M, abs)
    if abs.ndim == 3:
        m, l, k = abs.shape
        c = np.zeros((m,l,n))
        for i in tqdm(range(m)):
            for j in range(l):
                c[i,j,:], _ = scipy.optimize.nnls(M, abs[i,j,:])
    err = np.einsum("kn,...n->...k", M, c) - abs
    return c, err

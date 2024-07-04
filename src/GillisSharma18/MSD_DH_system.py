import numpy as np

def MSD_DH_system(m, c, k):
    """
    Generates a PH system as a linearization of a mass-spring-damper
    with mass vector m, damping vector c, and stiffness vector k.

    Parameters:
    m : array-like
        Mass vector.
    c : array-like
        Damping vector.
    k : array-like
        Stiffness vector.

    Returns:
    A : numpy.ndarray
        System matrix A.
    E : numpy.ndarray
        System matrix E.
    J : numpy.ndarray
        System matrix J.
    R : numpy.ndarray
        System matrix R.
    Q : numpy.ndarray
        System matrix Q.
    """
    n = len(m)
    M = np.zeros((n, n))
    K = np.zeros((n, n))
    D = np.zeros((n, n))

    for j in range(n - 1):
        M[j, j] = m[j]
        
        K[j, j] = k[j] + k[j + 1]
        K[j, j + 1] = -k[j + 1]
        K[j + 1, j] = K[j, j + 1]
        
        D[j, j] = c[j] + c[j + 1]
        D[j, j + 1] = -c[j + 1]
        D[j + 1, j] = D[j, j + 1]

    M[n - 1, n - 1] = m[n - 1]
    K[n - 1, n - 1] = k[n - 1]
    D[n - 1, n - 1] = c[n - 1]

    Z = np.zeros((n, n))
    J = np.block([[Z, -np.eye(n)], [np.eye(n), Z]])
    Q = np.block([[np.eye(n), Z], [Z, K]])
    R = np.block([[D, Z], [Z, Z]])
    A = (J - R) @ Q
    E = np.block([[M, Z], [Z, np.eye(n)]])

    return A, E, J, R, Q

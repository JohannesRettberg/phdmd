import logging

import numpy as np
from scipy.sparse import coo_matrix

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigvalsh, kron
from scipy.sparse import coo_matrix

from phdmd.linalg.definiteness import project_spsd


def hamiltonian_identification(X, Ham, project=False):
    """ """

    # Assemble data matrix
    n = X.shape[0]
    n_data = X.shape[1]
    min_data_points = n * (n + 1) // 2
    if n_data < min_data_points:
        logging.warning(
            f"There are not enough data points to identify Q from the Hamiltonian. It should be at least {min_data_points} but {n_data} data points were given."
        )

    XX = np.zeros((n_data, n**2))
    Dn = duplication_matrix(n)
    for i in range(n_data):
        XX[i, :] = np.squeeze(kron(X[:, i][:, np.newaxis], X[:, i][:, np.newaxis]))
    XX = XX @ Dn.toarray()

    # Solve for the symmetric part of Q
    Qvech = np.linalg.lstsq(XX, Ham, rcond=None)[0]
    Qvech = Dn.toarray() @ Qvech
    Q_id = Qvech.reshape(n, n)

    if project:
        # Project onto spsd matrices
        Q_id = project_spsd(Q_id)

    return Q_id


def duplication_matrix(n):
    """
    Create the duplication matrix of size n^2 x n(n+1)/2.

    Parameters:
    n (int): The dimension parameter.

    Returns:
    coo_matrix: The duplication matrix.
    """
    m = n * (n + 1) // 2
    nsq = n**2
    r = 0
    a = 0
    v = np.zeros(nsq, dtype=int)
    cn = np.cumsum(np.arange(n, 1, -1)) - 1

    for i in range(1, n + 1):
        if i > 1:
            v[r : r + i - 1] = i - n + cn[: i - 1]
            r += i - 1

        v[r : r + n - i + 1] = np.arange(a, a + n - i + 1)
        r += n - i + 1
        a += n - i + 1

        # print(v)
        # print(f"round {i}")
    rows = np.arange(nsq)
    cols = v
    data = np.ones(nsq)

    D = coo_matrix((data, (rows, cols)), shape=(nsq, m))
    return D


def test_hamiltonian_identification():
    # test case
    # Small script to test if the Hamiltonian identification problem is reasonable
    np.random.seed(0)  # for reproducibility

    example = "random"
    n = 5
    nrData = n**2

    # Create Q matrix
    if example == "diagonal":
        Q = np.diag(np.arange(1, n + 1))
    elif example == "msd":
        Q = np.array(
            [
                [4, 0, -4, 0, 0, 0],
                [0, 1 / 4, 0, 0, 0, 0],
                [-4, 0, 8, 0, -4, 0],
                [0, 0, 0, 1 / 4, 0, 0],
                [0, 0, -4, 0, 8, 0],
                [0, 0, 0, 0, 0, 1 / 4],
            ]
        )
        n = 6
    elif example == "random":
        Q = 3 * np.random.rand(n, n)
        Q = Q.T @ Q
    else:
        raise ValueError("example not known")

    # Information about the actual problem
    print(f"Example:\t\t\t {example}")
    print(f"Dimension:\t\t\t {n}")
    print(f"maximal training data:\t\t {nrData} (n^2 = {n**2})")
    print(f"minimal eigenvalue of Q:\t {min(eigvalsh(Q)):.3e}")

    # Create data
    X = np.random.rand(n, nrData)
    Ham = np.array([X[:, i].T @ Q @ X[:, i] for i in range(nrData)])

    # Test approximation quality over the data points
    if nrData > 50:
        # run at 8 data samples to save computation time
        ii = np.linspace(1, nrData + 1, 8, dtype=np.int32)
    else:
        # increase data samples by 2
        ii = np.arange(1, nrData + 1, 2)
    err = np.zeros(len(ii))
    errAfterProjection = np.zeros(len(ii))
    minEigVal = np.zeros(len(ii))
    minEigValAfterProjection = np.zeros(len(ii))

    for idx, i in enumerate(ii):
        # identify from Hamiltonian
        logging.info(f"Test Hamiltonian identification for {i} data samples")
        Q_id = hamiltonian_identification(X[:, :i], Ham[:i])
        # Performance measures
        minEigVal[idx] = min(eigvalsh(Q_id))
        err[idx] = np.linalg.norm(Q - Q_id) / np.linalg.norm(Q)

        Q_id_projected = hamiltonian_identification(X[:, :i], Ham[:i], project=True)
        minEigValAfterProjection[idx] = min(eigvalsh(Q_id_projected))
        errAfterProjection[idx] = np.linalg.norm(Q - Q_id_projected) / np.linalg.norm(Q)

    # Plot figures
    m = n * (n + 1) // 2
    fig, (ax1, ax2) = plt.subplots(2, 1)

    ax1.semilogy(ii, err, label="err")
    ax1.semilogy(ii, errAfterProjection, label="err after projection", linestyle="--")
    ax1.axvline(m, color="k", linestyle="--")
    ax1.set_ylabel("rel. error ||H-H_id||_F/||H||_F")
    ax1.legend()
    # ax1.set_aspect("equal", "box")

    ax2.plot(ii, minEigVal, label="min. eigenvalue")
    ax2.plot(
        ii,
        minEigValAfterProjection,
        label="min. eigenvalue after projection",
        linestyle="--",
    )
    ax2.axvline(m, color="k", linestyle="--")
    ax2.legend(loc="lower right")
    ax2.set_ylabel("min. eigenvalue")
    ax2.set_xlabel("n_data")
    # ax2.set_aspect("equal", "box")

    plt.savefig("hamIdentification.png")
    plt.show(block=False)

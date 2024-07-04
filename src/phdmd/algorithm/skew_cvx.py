import logging
import numpy as np

import cvxpy as cp


def skew_cvx(X, Y, n, J=None):
    """
    Solves the skew-symmetric Procrustes problem

        .. math::
            \min_{A} \|Y - AX\|_\mathrm{F} \quad s.t. \quad A = -A^T.

    n state dimension
    """
    # size of data
    n_data = X.shape[0]
    # number of inputs/outputs
    n_u = n_data - n

    G_inf = cp.Variable((n, n_u))
    N_inf = cp.Variable((n_u, n_u))

    if J is not None:
        # J known
        J_op = cp.bmat([[J, G_inf], [-G_inf.T, N_inf]])
    else:
        # J unknown
        J_inf = cp.Variable((n, n))
        J_op = cp.bmat([[J_inf, G_inf], [-G_inf.T, N_inf]])

    # skew-symmetric constraint
    constraints = [J_op == -J_op.T]

    # minimization problem
    minimize = cp.Minimize(cp.norm(Y - J_op @ X, "fro"))
    prob = cp.Problem(minimize, constraints)

    prob.solve()

    A = J_op.value

    e = np.linalg.norm(A @ X - Y, "fro")

    return A, e

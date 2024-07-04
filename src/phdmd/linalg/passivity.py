import numpy as np
import cvxpy as cp
from cvxpy.expressions.variable import Variable


def kyp_lmi(A, B, C, D, X, relaxed=False):
    """
    - Creates matrix function for KYP-LMI
    - Checks if KYP-LMI is fullfilled
    relaxed (bool): use relaxed KYP-LMI, see [GilliSharma18] (5.9)
    """
    # matrix function
    if isinstance(A, Variable):
        # use KYP for optimization
        # W = cp.bmat(
        #     [
        #         [-X @ A - A.conj().T @ X, C.conj().T - X @ B],
        #         [C - B.conj().T @ X, D + D.conj().T],
        #     ]
        # )
        W = cp.bmat(
            [
                [-X @ A - A.conj().T @ X, np.zeros((A.shape[0], D.shape[0]))],
                [np.zeros((D.shape[0], A.shape[0])), D + D.conj().T],
            ]
        )
        if relaxed:
            delta = cp.Variable()
            W = W + delta * np.eye(A.shape[0] + D.shape[0])
        return W
    else:
        W = np.array(
            [
                [-X @ A - A.conj().T @ X, C.conj().T - X @ B],
                [C - B.conj().T @ X, D + D.conj().T],
            ]
        )

        # check symmetric pos. definite
        try:
            np.linalg.cholesky(W)
            return W, True
        except np.linalg.LinAlgError:
            W += 1e-8 * np.eye(W.shape[0])
            try:
                np.linalg.cholesky(W)
                return W, True
            except np.linalg.LinAlgError:
                return W, False

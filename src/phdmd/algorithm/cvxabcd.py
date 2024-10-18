import logging
import os
import numpy as np

import cvxpy as cp

from pymor.models.iosys import LTIModel, PHLTIModel
from phdmd.linalg.passivity import kyp_lmi


def cvxabcd(
    X,
    Y,
    U,
    dXdt=None,
    delta_t=None,
    delta=1e-12,
    constraint_type="no",  # "KYP" | "no" | "nsd" | "nsd"
):
    """
    contraint_type (str): "KYP" | "no" | "nsd" | "nsd"

    """

    n = X.shape[0]
    n_u = U.shape[0]

    # data at mid timepoints + get dXdt
    if dXdt is None:
        assert delta_t is not None
        dXdt = (X[:, 1:] - X[:, :-1]) / delta_t
        X = 1 / 2 * (X[:, 1:] + X[:, :-1])
        U = 1 / 2 * (U[:, 1:] + U[:, :-1])
        Y = 1 / 2 * (Y[:, 1:] + Y[:, :-1])

    # instantiate operators
    A = cp.Variable((n, n))
    B = cp.Variable((n, n_u))
    C = cp.Variable((n_u, n))
    D = cp.Variable((n_u, n_u))

    # set up constraints
    epsilon = 1e-10  # enforce neg. def.
    constraints = []
    if constraint_type == "KYP":
        Q = cp.Variable((n, n), symmetric=True)
        constraints += [Q >> 0]
        W = kyp_lmi(A, B, C, D, Q, relaxed=False)
        constraints += [W >> 0]
    elif constraint_type == "KYP_relaxed":
        Q = cp.Variable((n, n), symmetric=True)
        constraints += [Q >> 0]
        W = kyp_lmi(A, B, C, D, Q, relaxed=True)
        constraints += [W >> 0]
    elif constraint_type == "nsd":
        constraints += [A << 0]
    elif constraint_type == "nsd+":
        constraints += [A + epsilon * np.eye(n) << 0]
    elif constraint_type == "no":
        # no constraints
        pass
    else:
        raise ValueError(f"Unkwon constraint option {constraint_type}.")

    # minimization problem
    # min_problem_state = cp.Minimize(cp.norm(dXdt - A @ X - B @ U, "fro"))
    # min_problem_output = cp.Minimize(cp.norm(Y - C @ X - D @ U, "fro"))
    data_lhs = np.vstack((dXdt, Y))
    data_rhs = np.vstack((X, U))
    if constraint_type == "KYP" or constraint_type == "KYP_relaxed":
        system_matrix_overall = cp.bmat([[A, B], [B.T @ Q, D]])
    else:
        system_matrix_overall = cp.bmat([[A, B], [C, D]])

    min_equation = cp.norm(data_lhs - system_matrix_overall @ data_rhs, "fro") ** 2
    if constraint_type == "KYP_relaxed":
        min_equation = min_equation + delta**2
    min_problem_overall = cp.Minimize(min_equation)

    # solve problem
    # problem_state = cp.Problem(min_problem_state, constraints)
    # problem_output = cp.Problem(min_problem_output)
    # problem_state.solve()
    # problem_output.solve()
    problem_overall = cp.Problem(min_problem_overall, constraints)
    problem_overall.solve(verbose=True)

    # check neg. def
    logging.info(f"Checking eigenvalues...")
    eig_vals_A = np.linalg.eigvals(A.value)
    logging.info(
        f"Maximum real-part of eigenvalue is negative: {all(np.real(eig_vals_A)<0)}. Max. EV is {np.max(np.real(eig_vals_A))}."
    )

    # create LTI model (pymor)
    lti_model = LTIModel.from_matrices(A.value, B.value, C.value, D.value)

    return lti_model



import logging
import os
import numpy as np

import cvxpy as cp

from pymor.models.iosys import LTIModel, PHLTIModel
from phdmd.utils.system import to_lti, to_phlti


def cvxph(
    X,
    Y,
    U,
    dXdt=None,
    delta_t=None,
    use_Berlin=True,
    J0=None,
    R0=None,
    H0=None,
    Q0=None,
    max_iter=20,
    delta=1e-12,
    use_cvx=False,
    J_known=None,
):

    # data at mid timepoints + get dXdt
    if dXdt is None:
        assert delta_t is not None
        dXdt = (X[:, 1:] - X[:, :-1]) / delta_t
        X = 1 / 2 * (X[:, 1:] + X[:, :-1])
        U = 1 / 2 * (U[:, 1:] + U[:, :-1])
        Y = 1 / 2 * (Y[:, 1:] + Y[:, :-1])

    n = X.shape[0]
    n_u = U.shape[0]

    # state equation level
    H_inf = cp.Variable((n, n), symmetric=True)
    R_inf = cp.Variable((n, n), symmetric=True)
    J_inf = cp.Variable((n, n))
    G_inf = cp.Variable((n, n_u))
    P_inf = cp.Variable((n, n_u))
    # infer N and S
    N_inf = cp.Variable((n_u, n_u))
    S_inf = cp.Variable((n_u, n_u), symmetric=True)

    # input-ouput equation level
    Z_data = cp.bmat([[H_inf @ dXdt], [-Y]])
    T_data = cp.bmat([[X], [U]])
    R_op = cp.bmat([[R_inf, P_inf], [P_inf.T, S_inf]])

    if J_known is not None:
        # J known
        assert np.allclose(J_known, -J_known.T)
        J_op = cp.bmat([[J_known, G_inf], [-G_inf.T, N_inf]])
    else:
        J_op = cp.bmat([[J_inf, G_inf], [-G_inf.T, N_inf]])

    epsilon = 1e-10
    constraints = [H_inf - epsilon * np.eye(n) >> 0]
    # constraints += [R_inf >> 0]
    # constraints += [S_inf >> 0]
    constraints += [J_op == -J_op.T]
    # constraints += [R_op >> 0]

    constraints += [R_op - epsilon * np.eye(n + n_u) >> 0]
    # minimize_state_level = cp.Minimize(cp.norm(H_inf@dXdt_train + R_inf@X_train - J@X_train - (G - P)@U_train,'fro'))
    minimize_IO_level = cp.Minimize(cp.norm(Z_data - (J_op - R_op) @ T_data, "fro"))
    prob = cp.Problem(minimize_IO_level, constraints)

    prob.solve()

    J_op_value = J_op.value
    R_op_value = R_op.value

    # check properties
    logging.info(
        f"J operator is skew-symmetric {np.allclose(J_op_value,-J_op_value.T)}"
    )
    logging.info(f"R operator is symmetric {np.allclose(R_op_value,R_op_value.T)}")
    R_eigvals = np.linalg.eigvals(R_op_value)
    logging.info(
        f"R operator pos. semi.def. {not(any(R_eigvals<=0))}. The smallest eigenvalue is {R_eigvals.min()}"
    )
    logging.info(f"H operator is symmetric {np.allclose(H_inf.value,H_inf.value.T)}")
    H_eigvals = np.linalg.eigvals(H_inf.value)
    logging.info(
        f"H operator pos. def. {not(any(H_eigvals<0))}. The smallest eigenvalue is {H_eigvals.min()}"
    )

    # logging.info(f"Norm for known operators {np.linalg.norm(E@dXdt_train - (J-R)@X_train - (G - P)@U_train,'fro')}")
    # logging.info(f"Norm for identified operators {np.linalg.norm(H_inf.value@dXdt_train - (J-R_inf.value)@X_train - (G - P)@U_train,'fro')}")

    # logging.info(
    #     f"Norm for known operators {np.linalg.norm(Z_data_orig -(J_op_orig - R_op_orig)@T_data_orig,'fro')}"
    # )
    logging.info(
        f"Norm for identified operators {np.linalg.norm(Z_data.value -(J_op_value - R_op_value)@T_data.value,'fro')}"
    )

    phlti = to_phlti(J_op_value, R_op_value, n=n, E=H_inf.value, no_feedtrough=True)

    return phlti

import numpy as np
import logging
from phdmd.data.generate import sim


def reproject(exp, V, method="implicit_midpoint", return_dXdt=False):
    logging.info(f"Reproject data.")
    # TODO: omit double code with discretize.py
    if isinstance(exp.u, list):
        U_is_list = True
        n_scenarios = len(exp.u)
        U_temp = exp.u.copy()[0]
        U_list = exp.u.copy()
    else:
        U_is_list = False
        U_temp = exp.u
        n_scenarios = 1

    if not isinstance(U_temp, np.ndarray):
        n_u = U_temp(exp.T).shape[0]
        if U_temp(exp.T).ndim < 2:
            n_u = 1
    else:
        n_u = U_temp.shape[0]

    if n_scenarios > 1:
        raise NotImplementedError(
            f"Reprojection for n_scenarios>1 is not implemented yet."
        )
    else:
        i_scenario = 0

    V = V.to_numpy().T

    X = np.zeros((V.shape[0], len(exp.T)))
    U = np.zeros((n_u, len(exp.T), n_scenarios))
    Y = np.zeros((n_u, len(exp.T), n_scenarios))
    x0 = exp.x0

    x0_red = V.T @ x0
    for k in range(len(exp.T) - 1):
        T = exp.T[k : k + 2]
        u = exp.u(T)
        U[:, k : k + 2, i_scenario], X[:, k : k + 2], Y[:, k : k + 2, i_scenario] = sim(
            exp.fom, u, T, V @ x0_red, method=method, return_dXdt=return_dXdt
        )
        x0_red = V.T @ X[:, -1]
    X_train_red = V.T @ X
    return U, X_train_red, Y

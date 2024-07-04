import logging

import numpy as np

from phdmd.algorithm.skew_procrustes import skew_procrustes
from phdmd.algorithm.skew_cvx import skew_cvx
from phdmd.linalg.definiteness import project_spsd, project_spd
from phdmd.linalg.symmetric import skew

from phdmd.utils.system import unstack

from matplotlib import pyplot as plt
from scipy.signal import savgol_filter


def phdmdh(
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
    r"""
    The pHDMD algorithm identifies a port-Hamiltonian system from state, output and input measurements.

    Define

        .. math::
            \begin{align*}
                \dmdW &\vcentcolon= \tfrac{1}{\timeStep}
                \begin{bmatrix}
                    \state_1-\state_0 & \ldots & \state_{\nrSnapshots} - \state_{\nrSnapshots-1}
                \end{bmatrix}\in\R^{\stateDim\times\nrSnapshots},\\
                \dmdV &\vcentcolon= \tfrac{1}{2}\begin{bmatrix}
                    \state_1+\state_0 & \ldots & \state_{\nrSnapshots} + \state_{\nrSnapshots-1}
                \end{bmatrix}\in\R^{\stateDim\times\nrSnapshots},\\
                \dmdU &\vcentcolon= \tfrac{1}{2}\begin{bmatrix}
                    \inpVar_1 + \inpVar_0 & \ldots & \inpVar_{\nrSnapshots} + \inpVar_{\nrSnapshots-1}
                \end{bmatrix} \in\R^{\inpVarDim\times\nrSnapshots},\\
                \dmdY &\vcentcolon= \tfrac{1}{2}\begin{bmatrix}
                    \outVar_1 + \outVar_0 & \ldots & \outVar_{\nrSnapshots} + \outVar_{\nrSnapshots-1}
                \end{bmatrix}\in\R^{\inpVarDim\times\nrSnapshots},
            \end{align*}

    and

        .. math::
            \dataZ \vcentcolon= \begin{bmatrix}H \dmdW \\ -\dmdY\end{bmatrix} \quad
            \mathrm{and} \quad \dataT \vcentcolon= \begin{bmatrix}\dmdV \\ \dmdU\end{bmatrix}.

    Solve the minimization problem

        .. math::
            \min_{\mathcal{J},\mathcal{R}} \|Z - (\mathcal{J} - \mathcal{R}) T\|_\mathrm{F},
            \quad \mathrm{s.t.} \quad \mathcal{J}=-\mathcal{J}^T, \mathcal{R}\in\mathbb{S}^{n}_{\succeq}.

    Parameters
    ----------
    X : numpy.ndarray
        Sequence of states.
    Y : numpy.ndarray
        Sequence of outputs.
    U : numpy.ndarray
        Sequence of inputs.
    dXdt: numpy.ndarray, optional
        Sequence of derivatives of the state. If not given the derivatives
        are approximated via implicit midpoints, then `delta_t` must be given.
    delta_t : float, optional
        Time step size. Only mandatory if `dXdt` is not given.
    H : numpy.ndarray, optional
        Hamiltonian matrix. If not given, `H` is assumed to be the identity.
    J0 : numpy.ndarray, optional
        Initial structure matrix. If not given `phdmd_init` is used for the initialization of `J` and `R`.
    R0 : numpy.ndarray
        Initial dissipation matrix. If not given `phdmd_init` is used for the initialization of `J` and `R`.
    max_iter : int, optional
        Maximum number of iterations. Default 20.
    delta : float, optional
        Convergence criteria. Default 1e-10.

    Returns
    -------
    J : numpy.ndarray
        Conservation of energy matrix.
    R : numpy.ndarray
        Dissipation matrix.
    """
    if dXdt is None:
        assert delta_t is not None
        dXdt = (X[:, 1:] - X[:, :-1]) / delta_t
        X = 1 / 2 * (X[:, 1:] + X[:, :-1])
        U = 1 / 2 * (U[:, 1:] + U[:, :-1])
        Y = 1 / 2 * (Y[:, 1:] + Y[:, :-1])

    filter_data = False
    if filter_data:
        dXdt = savgol_filter(dXdt, window_length=5, polyorder=3, deriv=0, delta=delta_t)

    # e = np.array([np.inf])
    # while e[-1] > 1e-3:

    logging.info("Perform pHDMD")
    if use_Berlin:
        assert H0 is not None
        H = H0
        Q = np.eye(X.shape[0])
    else:
        assert Q0 is not None
        Q = Q0
        H = np.eye(X.shape[0])

    T = np.concatenate((Q @ X, U))
    Z = np.concatenate((H @ dXdt, -Y))

    if J0 is None:
        assert R0 is None
        J, R = phdmd_init(T, Z)
    else:
        J = J0
        R = R0

    data = {
        "X": X,
        "dXdt": dXdt,
        "U": U,
        "Y": Y,
        "T": T,
        "Z": Z,
    }

    # G_known = np.array([[0.75], [0.5], [0.25], [0.0], [0.0], [0.0]])

    # P_known = np.array([[0.0], [0.0], [0.0], [0.0], [0.0], [0.0]])

    # J_op = np.block([[J_known, G_known], [-G_known.T, 0]])
    # R_op = np.block([[R_known, P_known], [P_known.T, 0]])

    # print(np.linalg.norm(Z - (J_op - R_op) @ T, "fro"))

    # pHDMD algorithm
    J, R, H, Q, e = phdmdh_FGM(
        data,
        J,
        R,
        H,
        Q,
        use_Berlin,
        max_iter,
        delta,
        use_cvx=use_cvx,
        n=X.shape[0],
        J_known=J_known,
    )

    return J, R, H, Q, e


def phdmd_init(T, Z, tol=1e-12):
    r"""
    Returns an initialization for the pHDMD algorithm by solving the related weighted minimization problem

        .. math::
            \min_{\mathcal{J},\mathcal{R}} \|T^TZ - T^T(\mathcal{J} - \mathcal{R}) T\|_\mathrm{F},
            \quad \mathrm{s.t.} \quad \mathcal{J}=-\mathcal{J}^T, \mathcal{R}\in\mathbb{S}^{n}_{\succeq}.

    Parameters
    ----------
    T : numpy.ndarray
        Stacked data matrix
    Z : numpy.ndarray
        Stacked data matrix
    tol : float, optional
        Zero tolerance.

    Returns
    -------
    J : numpy.ndarray
        Conservation of energy matrix.
    R : numpy.ndarray
        Dissipation matrix.
    """

    logging.info("pHDMD Initialization")

    n, m = T.shape

    U, s, Vh = np.linalg.svd(T, full_matrices=False)
    V = Vh.T

    r = np.argmax(s / s[0] < tol)
    r = r if r > 0 else len(s)
    s = s[:r]

    S = np.diag(s)
    S_inv = np.diag(1 / s)

    if r < n:
        logging.warning(f"Rank(T) < n + m ({r} < {n})")

    Z_1 = U.T @ Z @ V

    J_11 = skew(S @ Z_1[:r, :r])
    R_11 = project_spsd(-S @ Z_1[:r, :r])

    R = U[:, :r] @ S_inv @ R_11 @ S_inv @ U[:, :r].T
    J = U[:, :r] @ S_inv @ J_11 @ S_inv @ U[:, :r].T

    if r < n:
        # compensate rank deficit
        J_21 = np.linalg.lstsq(np.diag(s), Z_1[r:, :r].T, rcond=None)[0].T
        J_cmp = np.zeros((n, n))
        J_cmp[r:, :r] = J_21
        J_cmp[:r, r:] = -J_21.T
        J = J + U @ J_cmp @ U.T

    e = np.linalg.norm(T.T @ Z - T.T @ (J - R) @ T, "fro")
    logging.info(f"|T^T Z - T^T (J^(0) - R^(0)) T|_F = {e:.2e}")
    e_rel = e / np.linalg.norm(T.T @ Z)
    logging.info(f"|T^T Z - T^T (J^(0) - R^(0)) T|_F / |T^T Z|_F = {e_rel:.2e}")
    if r == n:
        c = np.linalg.norm(np.linalg.pinv(T.T), "fro")
        e0 = np.linalg.norm(Z - (J - R) @ T, "fro")
        logging.info(
            f"|Z - (J^(0) - R^(0)) T|_F = {e0:.2e} <= c |T^T Z - T^T (J^(0) - R^(0)) T|_F = {c * e:.2e}"
        )
        logging.info(f"with c = {c:.2e}")

    return J, R


def phdmdh_FGM(
    data,
    J0,
    R0,
    H0,
    Q0,
    use_Berlin,
    max_iter=20,
    delta=1e-12,
    use_cvx=False,
    n=None,
    J_known=None,
):
    r"""
    Iterative algorithm to solve the pHDMD problem via a fast-gradient method
    and the analytic solution of the skew-symmetric Procrustes problem.

    Parameters
    ----------
    T : numpy.ndarray
        Stacked data matrix.
    Z : numpy.ndarray
        Stacked data matrix.
    J0 : numpy.ndarray
        Initial matrix for `J`.
    R0 : numpy.ndarray
        Initial matrix for `R`.
    max_iter : int, optional
        Maximum number of iterations. Default 20.
    delta : float, optional
        Convergence criteria. Default 1e-10.

    Returns
    -------
    J : numpy.ndarray
        Conservation of energy matrix.
    R : numpy.ndarray
        Dissipation matrix.
    e : numpy.ndarray
        Value of the cost functional at each iteration.
    """

    logging.info("pHDMD Algorithm")

    X = data["X"]
    dXdt = data["dXdt"]
    T = data["T"]
    Z = data["Z"]
    U = data["U"]
    Y = data["Y"]

    R = R0
    J = J0
    H = H0
    Q = Q0

    # Precomputations
    n = X.shape[0]
    # Parameters and initialization
    alpha_0 = 0.1  # Parameter of the FGM in (0,1) - can be tuned.
    # for R
    TTt = T @ T.T
    wR, _ = np.linalg.eigh(TTt)
    LR = max(wR)  # Lipschitz constant
    muR = min(wR)
    qR = muR / LR
    betaR = np.zeros(max_iter)
    alphaR = np.zeros(max_iter + 1)
    alphaR[0] = alpha_0
    YrR = R
    # for H
    if use_Berlin:
        dXdt2T = dXdt @ dXdt.T
        XdXT = X @ dXdt.T
        UdXT = U @ dXdt.T
        wH, _ = np.linalg.eigh(dXdt2T)
        LH = max(wH)  # Lipschitz constant
        muH = min(wH)
        qH = muH / LH
        betaH = np.zeros(max_iter)
        alphaH = np.zeros(max_iter + 1)
        alphaH[0] = alpha_0
        YrH = H
    else:
        # for Q
        XXT = X @ X.T
        dXdtXT = dXdt @ X.T
        UXT = U @ X.T
        wQ, _ = np.linalg.eigh(XXT)
        LQ = max(wQ)  # Lipschitz constant
        muQ = min(wQ)
        qQ = muQ / LQ
        betaQ = np.zeros(max_iter)
        alphaQ = np.zeros(max_iter + 1)
        alphaQ[0] = alpha_0
        YrQ = Q

    e = np.zeros(max_iter + 1)

    # plt.figure()
    # plt.plot(X.T)
    # plt.savefig("X.png")
    # plt.figure()
    # plt.plot(dXdt.T)
    # plt.savefig("dXdt.png")

    e[0] = np.linalg.norm(Z - (J - R) @ T, "fro") / np.linalg.norm(Z)
    logging.info(f"|Z - (J^(0) - R^(0)) T|_F / |Z|_F = {e[0]:.2e}")

    for i in range(max_iter):
        # Previous iterate
        Rp = R
        Jp = J
        Hp = H
        Qp = Q

        Z_1 = Z + R @ T
        if use_cvx:
            J, _ = skew_cvx(T, Z_1, n, J=J_known)
        else:
            # Solution of the skew-symmetric Procrustes
            J, _ = skew_procrustes(T, Z_1)

        Z_2 = J @ T - Z
        # Projected gradient step from Y
        GR = YrR @ T @ T.T - Z_2 @ T.T
        R = project_spsd(YrR - GR / LR)

        # FGM Coefficients
        alphaR[i + 1] = (
            np.sqrt((alphaR[i] ** 2 - qR) ** 2 + 4 * alphaR[i] ** 2)
            + (qR - alphaR[i] ** 2)
        ) / 2
        betaR[i] = alphaR[i] * (1 - alphaR[i]) / (alphaR[i] ** 2 + alphaR[i + 1])

        # Linear combination of iterates

        YrR = R + betaR[i] * (R - Rp)

        # solve spds problem for H or Q
        no_feedtrough = True
        J_mat, G_mat, _, _ = unstack(J, n, no_feedtrough)
        R_mat, P_mat, _, _ = unstack(R, n, no_feedtrough)
        if use_Berlin:
            GH = YrH @ dXdt2T - (J_mat - R_mat) @ XdXT - (G_mat - P_mat) @ UdXT
            H = project_spd(YrH - GH / LH)
            add_regularization = False
            if add_regularization:
                eig_vals, eig_vecs = np.linalg.eig(H)
                if min(eig_vals) < 1e-10:
                    H += 1e-10 * np.eye(H.shape[0])
                    logging.info(
                        f"added regularizing diag entries to H. Mininum eigenvalue is {min(eig_vals)}"
                    )

            # FGM Coefficients
            alphaH[i + 1] = (
                np.sqrt((alphaH[i] ** 2 - qH) ** 2 + 4 * alphaH[i] ** 2)
                + (qH - alphaH[i] ** 2)
            ) / 2
            betaH[i] = alphaH[i] * (1 - alphaH[i]) / (alphaH[i] ** 2 + alphaH[i + 1])
            # Linear combination of iterates
            YrH = H + betaH[i] * (H - Hp)
            Z = np.concatenate((H @ dXdt, -Y))
        else:
            GQ = YrQ @ XXT - (
                np.linalg.solve((J_mat - R_mat), dXdtXT - (G_mat - P_mat) @ UXT)
            )
            Q = project_spd(YrQ - GQ / LQ)
            add_regularization = False
            if add_regularization:
                eig_vals, eig_vecs = np.linalg.eig(Q)
                if min(eig_vals) < 1e-10:
                    Q += 1e-10 * np.eye(Q.shape[0])
                    logging.info(
                        f"added regularizing diag entries to Q. Mininum eigenvalue is {min(eig_vals)}"
                    )

            # FGM Coefficients
            alphaQ[i + 1] = (
                np.sqrt((alphaQ[i] ** 2 - qQ) ** 2 + 4 * alphaQ[i] ** 2)
                + (qQ - alphaQ[i] ** 2)
            ) / 2
            betaQ[i] = alphaQ[i] * (1 - alphaQ[i]) / (alphaQ[i] ** 2 + alphaQ[i + 1])
            # Linear combination of iterates
            YrQ = Q + betaQ[i] * (Q - Qp)
            T = np.concatenate((Q @ X, U))

            # Recalculate T-parameters?????
            # wR, _ = np.linalg.eigh(T@T.T)
            # LR = max(wR)  # Lipschitz constant
            # muR = min(wR)
            # qR = muR / LR

        e[i + 1] = np.linalg.norm(Z - (J - R) @ T, "fro") / np.linalg.norm(Z)
        logging.info(f"|Z - (J^({i + 1}) - R^({i + 1})) T|_F / |Z|_F = {e[i + 1]:.2e}")

        eps = (
            np.linalg.norm(Jp - J, "fro") / (np.linalg.norm(J, "fro"))
            + np.linalg.norm(Rp - R, "fro") / (np.linalg.norm(R, "fro"))
            + np.linalg.norm(Hp - H, "fro") / (np.linalg.norm(H, "fro"))
            + np.linalg.norm(Qp - Q, "fro") / (np.linalg.norm(Q, "fro"))
        )
        if eps < delta or np.abs(e[i + 1] - e[i]) < delta:
            e = e[: i + 2]
            logging.info(f"Converged after {i + 1} iterations.")
            break

        if i == max_iter - 1:
            logging.info(
                f"PHDMDH has not converged. It has reached the max_iter value {max_iter}."
            )

    return J, R, H, Q, e

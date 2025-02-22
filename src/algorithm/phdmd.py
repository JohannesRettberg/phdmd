import logging

import numpy as np

from algorithm.skew_procrustes import skew_procrustes
from linalg.definiteness import project_spsd
from linalg.symmetric import skew


def phdmd(X, Y, U, dXdt=None, delta_t=None, H=None, J0=None, R0=None, max_iter=5, delta=1e-12):
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

    if H is None:
        H = np.eye(X.shape[0])

    T = np.concatenate((X, U))
    Z = np.concatenate((H @ dXdt, -Y))

    logging.info('Perform pHDMD')

    # pHDMD initialization
    if J0 is None:
        assert R0 is None
        J, R = phdmd_init(T, Z)
    else:
        J = J0
        R = R0

    # pHDMD algorithm
    J, R, e = phdmd_FGM(T, Z, J, R, max_iter, delta)

    return J, R, e


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
        logging.warning(f'Rank(T) < n + m ({r} < {n})')

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

    e = np.linalg.norm(T.T @ Z - T.T @ (J - R) @ T, 'fro')
    logging.info(f'|T^T Z - T^T (J^(0) - R^(0)) T|_F = {e:.2e}')
    e_rel = e / np.linalg.norm(T.T @ Z)
    logging.info(f'|T^T Z - T^T (J^(0) - R^(0)) T|_F / |T^T Z|_F = {e_rel:.2e}')
    if r == n:
        c = np.linalg.norm(np.linalg.pinv(T.T), 'fro')
        e0 = np.linalg.norm(Z - (J - R) @ T, 'fro')
        logging.info(f'|Z - (J^(0) - R^(0)) T|_F = {e0:.2e} <= c |T^T Z - T^T (J^(0) - R^(0)) T|_F = {c * e:.2e}')
        logging.info(f'with c = {c:.2e}')

    return J, R


def phdmd_FGM(T, Z, J0, R0, max_iter=20, delta=1e-12):
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

    R = R0
    J = J0

    # Precomputations
    TTt = T @ T.T
    w, _ = np.linalg.eigh(TTt)
    L = max(w)  # Lipschitz constant
    mu = min(w)
    q = mu / L

    beta = np.zeros(max_iter)
    alpha = np.zeros(max_iter + 1)
    e = np.zeros(max_iter + 1)

    # Parameters and initialization
    alpha_0 = 0.1  # Parameter of the FGM in (0,1) - can be tuned.

    Yr = R
    alpha[0] = alpha_0
    e[0] = np.linalg.norm(Z - (J - R) @ T, 'fro') / np.linalg.norm(Z)
    logging.info(f'|Z - (J^(0) - R^(0)) T|_F / |Z|_F = {e[0]:.2e}')

    for i in range(max_iter):
        # Previous iterate
        Rp = R
        Jp = J

        Z_1 = Z + R @ T
        # Solution of the skew-symmetric Procrustes
        J, _ = skew_procrustes(T, Z_1)

        Z_2 = J @ T - Z
        # Projected gradient step from Y
        G = Yr @ TTt - Z_2 @ T.T
        R = project_spsd(Yr - G / L)

        # FGM Coefficients
        alpha[i + 1] = (np.sqrt((alpha[i] ** 2 - q) ** 2 + 4 * alpha[i] ** 2) + (q - alpha[i] ** 2)) / 2
        beta[i] = alpha[i] * (1 - alpha[i]) / (alpha[i] ** 2 + alpha[i + 1])

        # Linear combination of iterates
        Yr = R + beta[i] * (R - Rp)

        e[i + 1] = np.linalg.norm(Z - (J - R) @ T, 'fro') / np.linalg.norm(Z)
        logging.info(f'|Z - (J^({i + 1}) - R^({i + 1})) T|_F / |Z|_F = {e[i + 1]:.2e}')

        eps = np.linalg.norm(Jp - J, 'fro') / (np.linalg.norm(J, 'fro')) + \
              np.linalg.norm(Rp - R, 'fro') / (np.linalg.norm(R, 'fro'))
        if eps < delta or np.abs(e[i + 1] - e[i]) < delta:
            e = e[:i + 2]
            logging.info(f'Converged after {i + 1} iterations.')
            break

    return J, R, e

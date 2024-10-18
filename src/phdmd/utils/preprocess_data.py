import numpy as np
import scipy
import logging
import control as ct

from pymor.models.iosys import PHLTIModel, LTIModel
from pymor.vectorarrays.numpy import NumpyVectorSpace
from pymor.operators.numpy import NumpyMatrixOperator
from pymor.reductors.basic import LTIPGReductor
from pymor.basic import project
from pymor.algorithms.to_matrix import to_matrix

from phdmd.linalg.definiteness import project_spsd, project_spd
from phdmd.algorithm.hamiltonian_identification import hamiltonian_identification


def get_Riccati_transform(fom, get_phlti=False):
    A, B, C, D, E = fom.to_abcde_matrices()
    # add regularization to D to be nonsingular
    D = D + np.eye(D.shape[0]) * 1e-12
    lti_model = LTIModel.from_matrices(A, B, C, D, E)

    # %% Check for positive-real systems
    calT = lambda s: D + C @ np.linalg.inv(s * np.eye(A.shape[0]) - A) @ B
    Phi = lambda s: (calT(-s)).conj().T + calT(s)
    # Transfer function Phi needs to be positive semidefinite for all omega in R (checked for several points)
    omega_vec = np.linspace(0.1, 1000, 100)
    negEigValPhi = False
    omega_neg_eig_list = []
    for omega in omega_vec:
        eig_vals_Phi, _ = np.linalg.eig(Phi(1j * omega))
        if (eig_vals_Phi < 0).any():
            omega_neg_eig_list.append(omega)
            # print(f"negative eigenvalues at i{omega} rad/s \n")
            negEigValPhi = True
    if not negEigValPhi:
        logging.info(
            f"Phi is positive semidefinite - X that satisfies W(X) should exist"
        )
    else:
        logging.info(
            f"There are negative eigenvalues at {len(omega_neg_eig_list)} frequency points out of {len(omega_vec)} tested points."
            f"These points are {omega_neg_eig_list}."
        )
        logging.warning(
            f"Phi is NOT positive semidefinite - X that satisfies W(X) should NOT exist"
        )

    # check passivity
    if E is not None:
        if not (np.allclose(np.eye(E.shape[0]), E)):
            # E not identity
            logging.info(
                f"Transform E to identity to check passivity with control systems library."
            )
            # transform ss system for passivity check
            A_invE = np.linalg.solve(E, A)
            B_invE = np.linalg.solve(E, B)
            ct_lti_system = ct.StateSpace(A_invE, B_invE, C, D)
        else:
            # E identity
            ct_lti_system = ct.StateSpace(A, B, C, D)
    else:
        # E None
        ct_lti_system = ct.StateSpace(A, B, C, D)

    is_passive = ct_lti_system.ispassive()
    logging.info(f"System is passive: {is_passive}.")
    if is_passive:
        logging.info(f"System is passive.")
    else:
        logging.info("System might be passive. Only sufficient condition tested.")

    # get ph model
    try:
        # %% transform into pH system
        logging.info(
            f"Trying to transform state-space system into pH system through Riccati solution..."
        )
        phlti_model, X = PHLTIModel.from_passive_LTIModel(lti_model)
        # X = lti_model.gramian("pr_o_dense")
        logging.info(f"PH system successfully transformed!")
    except:
        logging.error(f"PH model could not be calculated. ")
        phlti_model, X = None, None

    # get transformation matrix
    try:
        T = np.linalg.cholesky(X).conj().T  # transformation matrix
    except:
        T = None
        logging.error(f"Transformation matrix could not be calculated.")

    if get_phlti:
        return T, phlti_model
    else:
        return T


def get_initial_energy_matrix(exp, additional_data=None):

    HQ_init_strat = exp.HQ_init_strat

    if exp.use_Berlin:
        if HQ_init_strat == "id":
            logging.info("Identity matrix is used as H initialization.")
            H = np.eye(exp.H.shape[0])
        elif HQ_init_strat == "rand":
            logging.info("Random spd H0 matrix is used as H initialization.")
            H = np.random.rand(exp.H.shape[0], exp.H.shape[0])
            H = project_spd(H)
        elif HQ_init_strat == "known":
            logging.info("Known H0 matrix is used as H initialization.")
            H = exp.H
        elif HQ_init_strat == "Ham":
            logging.info(f"Calculating initial energy matrix from Hamiltonian.")
            X = additional_data["X"]
            project_bool = additional_data["project"]
            red_Ham_treshold = 200
            if X.shape[0] > red_Ham_treshold:
                # calculation of n^2 not possible
                VV, S, _ = np.linalg.svd(X, full_matrices=False)
                # use 200 modes
                V = NumpyVectorSpace.from_numpy(VV[:, :red_Ham_treshold].T, id="STATE")
                # reduced data
                X = to_matrix(
                    project(NumpyMatrixOperator(X, range_id="STATE"), V, None)
                )
                H = to_matrix(
                    project(
                        NumpyMatrixOperator(exp.H, source_id="STATE", range_id="STATE"),
                        V,
                        V,
                    )
                )
            else:
                H = exp.H
            Ham = np.array([X[:, i].T @ H @ X[:, i] for i in range(X.shape[1])])
            H = hamiltonian_identification(X, Ham, project_bool)

            if X.shape[0] > red_Ham_treshold:
                # reproject to high-dimensional space
                H = V.to_numpy().T @ H @ V.to_numpy()
        else:
            raise ValueError(f"Unknown value of HQ_init_strat {HQ_init_strat}")
        Q = None
    else:
        if HQ_init_strat == "id":
            logging.info("Identity matrix is used as Q initialization.")
            Q = np.eye(exp.Q.shape[0])
        elif HQ_init_strat == "rand":
            logging.info("Random spd Q0 matrix is used as Q initialization.")
            Q = np.random.rand(exp.Q.shape[0], exp.Q.shape[0])
            Q = project_spd(Q)
        elif HQ_init_strat == "known":
            logging.info("Known Q0 matrix is used as Q initialization.")
            Q = exp.Q
        elif HQ_init_strat == "Ham":
            logging.info(f"Calculating initial energy matrix from Hamiltonian.")
            X = additional_data["X"]
            project_bool = additional_data["project"]
            red_Ham_treshold = 200
            if X.shape[0] > red_Ham_treshold:
                # calculation of n^2 not possible
                VV, S, _ = np.linalg.svd(X, full_matrices=False)
                # use 200 modes
                V = NumpyVectorSpace.from_numpy(VV[:, :red_Ham_treshold].T, id="STATE")
                # reduced data
                X = to_matrix(
                    project(NumpyMatrixOperator(X, range_id="STATE"), V, None)
                )
                Q = to_matrix(
                    project(
                        NumpyMatrixOperator(exp.Q, source_id="STATE", range_id="STATE"),
                        V,
                        V,
                    )
                )
            else:
                Q = exp.Q
            Ham = np.array([X[:, i].T @ exp.Q @ X[:, i] for i in range(X.shape[1])])
            Q = hamiltonian_identification(X, Ham, project_bool)
            if X.shape[0] > red_Ham_treshold:
                # reproject to high-dimensional space
                Q = V.to_numpy().T @ Q @ V.to_numpy()
        else:
            raise ValueError(f"Unknown value of HQ_init_strat {HQ_init_strat}")
        H = None

    return H, Q


def perturb_initial_energy_matrix(exp, H, Q, min_eig_value=1e-14):
    logging.info("Energy matrix is perturbed.")

    if isinstance(exp.perturb_value, str) and exp.perturb_value == "last_sing":
        # perturb by the smallest singular value
        if Q is None:
            if scipy.sparse.issparse(H):
                U_perturb, S_perturb, Vh_perturb = scipy.sparse.linalg.svds(H)
            else:
                U_perturb, S_perturb, Vh_perturb = np.linalg.svd(H)
            # matrix for rank deficient perturbation:
            # perturbation_matrix = -S_perturb[-1]*(U_perturb[:,-1][:,np.newaxis] @ Vh_perturb[-1,:][np.newaxis,:])

            logging.info(f"Perturbing with last singular value {S_perturb[-1]}")
            rand_matrix = np.random.random((H.shape[0], H.shape[0]))
            perturbation_matrix = (
                S_perturb[-1] * rand_matrix / np.linalg.norm(rand_matrix)
            )
        if H is None:
            if scipy.sparse.issparse(Q):
                U_perturb, S_perturb, Vh_perturb = scipy.sparse.linalg.svds(Q)
            else:
                U_perturb, S_perturb, Vh_perturb = np.linalg.svd(Q)
            # matrix for rank deficient perturbation:
            # perturbation_matrix = -S_perturb[-1]*(U_perturb[:,-1][:,np.newaxis] @ Vh_perturb[-1,:][np.newaxis,:])

            logging.info(f"Perturbing with last singular value {S_perturb[-1]}")
            rand_matrix = np.random.random((Q.shape[0], Q.shape[0]))
            perturbation_matrix = (
                S_perturb[-1] * rand_matrix / np.linalg.norm(rand_matrix)
            )
    else:
        # perturbation with small epsilon value
        if Q is None:
            rand_matrix = np.random.random((H.shape[0], H.shape[0]))
        if H is None:
            rand_matrix = np.random.random((Q.shape[0], Q.shape[0]))
        perturbation_matrix = (
            exp.perturb_value * rand_matrix / np.linalg.norm(rand_matrix)
        )

    logging.info(f"Norm of perturbation matrix {np.linalg.norm(perturbation_matrix)}")
    if Q is None:
        H = H + perturbation_matrix
        H = project_spd(H, min_eig_value=min_eig_value)
    if H is None:
        Q = Q + perturbation_matrix
        Q = project_spd(Q, min_eig_value=min_eig_value)

    return H, Q

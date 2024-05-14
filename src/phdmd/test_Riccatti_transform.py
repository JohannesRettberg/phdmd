import logging
import os
import numpy as np
import scipy

from phdmd import config
from phdmd.data.generate import generate, sim
from phdmd.evaluation.evaluation import evaluate
from phdmd.linalg.definiteness import project_spsd, project_spd
from pymor.algorithms.to_matrix import to_matrix

from pymor.models.iosys import PHLTIModel, LTIModel

from matplotlib import pyplot as plt


def main():
    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)

    if not os.path.exists(config.simulations_path):
        os.makedirs(config.simulations_path)

    if not os.path.exists(config.evaluations_path):
        os.makedirs(config.evaluations_path)

    # specify experiments in config.py
    experiments = config.experiments

    for exp in experiments:
        logging.info(f'Experiment: {exp.name}')
        lti_dict = {}

        if exp.use_Riccatti_transform:
            A, B, C, D, E = exp.fom.to_abcde_matrices()

            if not np.allclose(E,np.eye(E.shape[0])):
                raise ValueError(f"E needs to be the identity.")

            # add regularization for D to be nonsingular
            D = D + np.eye(D.shape[0]) * 1e-12
            lti_model = LTIModel.from_matrices(A, B, C, D, E, solver_options={'ricc_pos_dense': 'scipy'}) # {'ricc_pos_lrcf': 'slycot'} or {'ricc_pos_dense': 'slycot'}
            phlti_model_passive, X_test = PHLTIModel.from_passive_LTIModel(lti_model)
            X = lti_model.gramian('pr_o_dense')
            # X = X.to_numpy()
            T = scipy.linalg.cholesky(X) # transformation matrix
            # T = scipy.linalg.sqrtm(X)
            logging.info(f"Factorization of X works: {np.allclose(X,T.conj().T@T)}")

            # matrix function
            W = lambda X: np.block(
            [
                [-X @ A - A.conj().T @ X, C.conj().T - X @ B],
                [C - B.conj().T @ X, D + D.conj().T],
            ]
                )
            
            # Check KYP-LMI
             # check KYP inequality
            eig_vals_X, eig_vectors_X = np.linalg.eig(X)
            if (eig_vals_X <= 0).any():
                print(f"The solution X is NOT pos. def.")
            else:
                print(f"Nice! The solution X is pos. def.")
            eig_vals_W, eig_vectors_W = np.linalg.eig(W(X))
            # eps = 1e-10
            if (eig_vals_W < 0).any():
                print(
                    f"WARNING! The KYP inequality is not satisfied. Minimum eigenvalue: {eig_vals_W.min()}"
                )
            else:
                print(f"Nice! The KYP inequality is satisfied.")


            # leads to a transformed state-space in T-coordinates
            # T_inv = np.linalg.inv(T)
            # A_T = T @ A @ T_inv
            # B_T = T @ B
            # C_T = C @ T_inv

            # # we obtain a pH representation in T-coordinates
            # J_T = 1 / 2 * (A_T - A_T.conj().T)
            # R_T = -1 / 2 * (A_T + A_T.conj().T)
            # K_T = 1 / 2 * (C_T.conj().T - B_T)
            # G_T = 1 / 2 * (C_T.conj().T + B_T)

            # transformed lti system
            T_inv = np.linalg.inv(T)
            A_T = T@A@T_inv
            B_T = T@B
            C_T = C@T_inv
            D_T = D
            J_T = 0.5*(A_T - A_T.conj().T)
            R_T = -0.5*(A_T + A_T.conj().T)
            K_T = 0.5*(C_T.conj().T-B_T)
            G_T = 0.5*(C_T.conj().T+B_T)
            Q_T = np.eye(A_T.shape[0])
            E_T = E
            phlti_model_Qeye = PHLTIModel.from_matrices(J = J_T, R = R_T, G = G_T, P = K_T, S = D_T, N=None, E=E_T, Q=Q_T)

        # Non-transformed data
        # Generate/Load training data
        U_train, X_train, Y_train = sim(exp.fom, exp.u, exp.T, exp.x0, method=exp.time_stepper)

        # Results from_passive_LTI function
        U_passive_lti, X_passive_lti, Y_passive_lti = sim(phlti_model_passive, exp.u, exp.T, exp.x0, method=exp.time_stepper)

        # Results transformed Q=I model
        U_Qeye, X_Qeye, Y_Qeye = sim(phlti_model_Qeye, exp.u, exp.T, T@exp.x0, method=exp.time_stepper)

        # output
        plt.figure()
        plt.plot(exp.T,Y_train.T,label="orig")
        plt.plot(exp.T,Y_passive_lti.T,'--',label="from_passive")
        plt.plot(exp.T,Y_Qeye.T,'-.',label="Qeye")
        plt.legend()
        plt.title("Output")
        plt.xlabel("Time [s]")
        plt.savefig("Riccatti_traj_output_comparision.png")

        # state
        plt.figure()
        plt.plot(exp.T,X_train.T,label="orig")
        plt.plot(exp.T,X_passive_lti.T,'--',label="from_passive")
        plt.plot(exp.T,(T_inv@X_Qeye).T,'-.',label="Qeye")
        plt.legend()
        plt.title("State")
        plt.xlabel("Time [s]")
        plt.savefig("Riccatti_traj_state_comparision.png")


if __name__ == "__main__":
    main()

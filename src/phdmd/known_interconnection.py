import logging
import os
import numpy as np

from phdmd import config
from phdmd.data.generate import generate
from phdmd.evaluation.evaluation import evaluate
from phdmd.linalg.definiteness import project_spsd, project_spd
from phdmd.utils.system import to_lti, to_phlti

from pymor.models.iosys import PHLTIModel, LTIModel

import cvxpy as cp


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
        # Generate/Load training data
        X_train, Y_train, U_train = generate(exp)
        dXdt_train = (X_train[:, 1:] - X_train[:, :-1]) / exp.delta
        X_train = 1 / 2 * (X_train[:, 1:] + X_train[:, :-1])
        U_train = 1 / 2 * (U_train[:, 1:] + U_train[:, :-1])
        Y_train = 1 / 2 * (Y_train[:, 1:] + Y_train[:, :-1])

        n = exp.fom.order
        n_u = U_train.shape[0]
        J, R, G, P, S, N, E, Q = exp.fom.to_matrices()

        # original operators
        Z_data_orig = np.block([[E@dXdt_train],[-Y_train]])
        T_data_orig = np.block([[X_train],[U_train]])
        J_op_orig = np.block([[J, G],[-G.T, N]])
        R_op_orig = np.block([[R, P],[P.T, S]])

        # state equation level
        H_inf = cp.Variable((n,n), symmetric=True)
        R_inf = cp.Variable((n,n), symmetric=True)
        J_inf = cp.Variable((n,n))
        G_inf = cp.Variable((n,n_u))
        P_inf = cp.Variable((n,n_u))
        # infer N and S
        N_inf = cp.Variable((n_u,n_u))
        S_inf = cp.Variable((n_u,n_u), symmetric=True)
        # known that feedback is zero
        N_inf = np.array([0])[:,np.newaxis]
        S_inf = np.array([0])[:,np.newaxis]

        # input-ouput equation level
        Z_data = cp.bmat([[H_inf@dXdt_train],[-Y_train]])
        T_data = cp.bmat([[X_train],[U_train]])        
        R_op = cp.bmat([[R_inf, P_inf],[P_inf.T, S_inf]])

        J_is_known = False
        if J_is_known:
            # J known
            J_op = cp.bmat([[J, G_inf],[-G_inf.T, N_inf]])  
        else:
            J_op = cp.bmat([[J_inf, G_inf],[-G_inf.T, N_inf]])  

        constraints = [H_inf >> 0]
        # constraints += [R_inf >> 0]
        # constraints += [S_inf >> 0]
        constraints += [J_op == -J_op.T]
        # constraints += [R_op >> 0]
        epsilon = 1e-10
        constraints += [R_op - epsilon*np.eye(n + n_u) >> 0]
        minimize_state_level = cp.Minimize(cp.norm(H_inf@dXdt_train + R_inf@X_train - J@X_train - (G - P)@U_train,'fro'))
        minimize_IO_level = cp.Minimize(cp.norm(Z_data -(J_op - R_op)@T_data,'fro'))
        prob = cp.Problem(minimize_IO_level, constraints)

        prob.solve()

        J_op_value = J_op.value
        R_op_value = R_op.value

        # check properties
        logging.info(f"J operator is skew-symmetric {np.allclose(J_op_value,-J_op_value.T)}")
        logging.info(f"R operator is symmetric {np.allclose(R_op_value,R_op_value.T)}")
        R_eigvals = np.linalg.eigvals(R_op_value)
        logging.info(f"R operator pos. semi.def. {not(any(R_eigvals<=0))}. The smallest eigenvalue is {R_eigvals.min()}")
        logging.info(f"H operator is symmetric {np.allclose(H_inf.value,H_inf.value.T)}")
        H_eigvals = np.linalg.eigvals(H_inf.value)
        logging.info(f"H operator pos. def. {not(any(H_eigvals<0))}. The smallest eigenvalue is {H_eigvals.min()}")

        # logging.info(f"Norm for known operators {np.linalg.norm(E@dXdt_train - (J-R)@X_train - (G - P)@U_train,'fro')}")
        # logging.info(f"Norm for identified operators {np.linalg.norm(H_inf.value@dXdt_train - (J-R_inf.value)@X_train - (G - P)@U_train,'fro')}")

        logging.info(f"Norm for known operators {np.linalg.norm(Z_data_orig -(J_op_orig - R_op_orig)@T_data_orig,'fro')}")
        logging.info(f"Norm for identified operators {np.linalg.norm(Z_data.value -(J_op_value - R_op_value)@T_data.value,'fro')}")

        phlti = to_phlti(J_op_value, R_op_value, n=n, E=H_inf.value, no_feedtrough=True)
        
        lti_dict = {
            "Original": exp.fom,
            "cvx_phlti": phlti,
        }
        evaluate(exp, lti_dict, compute_hinf=False)



if __name__ == "__main__":
    main()

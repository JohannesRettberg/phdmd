import numpy as np
import logging

from pymor.models.iosys import PHLTIModel, LTIModel
from phdmd.linalg.definiteness import project_spsd, project_spd

def get_Riccati_transform(exp):
    A, B, C, D, E = exp.fom.to_abcde_matrices()
    # add regularization to D to be nonsingular
    D = D + np.eye(D.shape[0]) * 1e-12
    lti_model = LTIModel.from_matrices(A, B, C, D, E)
    # phlti_model = PHLTIModel.from_passive_LTIModel(lti_model)
    X = lti_model.gramian('pr_o_dense')
    T = np.linalg.cholesky(X).conj().T # transformation matrix

    return T

def get_initial_energy_matrix(exp):


    HQ_init_strat = exp.HQ_init_strat


    if exp.use_Berlin:
        if HQ_init_strat == 'id':
            logging.info("Identity matrix is used as H initialization.")
            H = np.eye(exp.H.shape[0])
        elif HQ_init_strat == 'rand':
            logging.info("Random spd H0 matrix is used as H initialization.")
            H = np.random.rand(exp.H.shape[0],exp.H.shape[0])
            H = project_spd(H)            
        elif HQ_init_strat == 'known':
            logging.info("Known H0 matrix is used as H initialization.")
            H = exp.H
        Q = None
    else:
        if HQ_init_strat == 'id':
            logging.info("Identity matrix is used as Q initialization.")
            Q = np.eye(exp.Q.shape[0])
        elif HQ_init_strat == 'rand':
            logging.info("Random spd Q0 matrix is used as Q initialization.")
            Q = np.random.rand(exp.Q.shape[0],exp.Q.shape[0])
            Q = project_spd(Q) 
            pass
        elif HQ_init_strat == 'known':
            logging.info("Known Q0 matrix is used as Q initialization.")
            Q = exp.Q
        H = None

    return H, Q

def perturb_initial_energy_matrix(exp,H,Q):
    logging.info("Energy matrix is perturbed.")
    # perturb by the smallest singular value
    if Q is None:
        U_perturb, S_perturb, Vh_perturb = np.linalg.svd(H)
        # matrix for rank deficient perturbation:
        # perturbation_matrix = -S_perturb[-1]*(U_perturb[:,-1][:,np.newaxis] @ Vh_perturb[-1,:][np.newaxis,:]) 
        
        if isinstance(exp.perturb_value,str) and exp.perturb_value == 'last_sing':
            # perturbation with smallest singular value 
            logging.info(f"Perturbing with last singular value {S_perturb[-1]}")
            perturbation_matrix = S_perturb[-1]*np.random.random((H.shape[0],H.shape[0]))
        else:
            # perturbation with small epsilon value
            perturbation_matrix = exp.perturb_value*np.random.random((H.shape[0],H.shape[0]))

        H = H + perturbation_matrix
        H = project_spd(H) 
    if H is None:
        U_perturb, S_perturb, Vh_perturb = np.linalg.svd(Q)
        # matrix for rank deficient perturbation:
        # perturbation_matrix = -S_perturb[-1]*(U_perturb[:,-1][:,np.newaxis] @ Vh_perturb[-1,:][np.newaxis,:]) 
        
        if isinstance(exp.perturb_value,str) and exp.perturb_value == 'last_sing':
            # perturbation with smallest singular value 
            logging.info(f"Perturbing with last singular value {S_perturb[-1]}")
            perturbation_matrix = S_perturb[-1]*np.random.random((Q.shape[0],Q.shape[0]))
        else:
            # perturbation with small epsilon value
            perturbation_matrix = exp.perturb_value*np.random.random((Q.shape[0],Q.shape[0]))
        Q = Q + perturbation_matrix
        Q = project_spd(Q)   
    logging.info(f"Norm of perturbation matrix {np.linalg.norm(perturbation_matrix)}")   

    return H, Q  